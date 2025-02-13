from algorithm import get_minimal_route
from generate_tsp import generate_tsp
from mutate_tsp import apply_mutation
import numpy as np
import json
import os
import glob
import time

# debugging
from icecream import ic

# # bash job
import argparse
import ast 

# Added some extra arguments to Wouter's experiment
# Initialize the parser
parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

# Add arguments
parser.add_argument('sizes', type=str, help='A list of city sizes, e.g., "[10,12]"')
parser.add_argument('ranges', type=str, help='A list of value ranges, e.g., "[10,1000]"')
parser.add_argument('mutations', type=int, help='An integer number of mutations, e.g., 500')
parser.add_argument('continuation', type=str, default="", nargs='?', help='A list of matrix continuations, e.g., "[(7,10),(50,10)]". Corresponding matrices must be in "Progress" folder') # tuple of city size and range
parser.add_argument('--tsp_type', type=str, choices=['euclidean', 'asymmetric'], required=True, help='Type of TSP to generate: symmetric or asymmetric.')
parser.add_argument('--distribution', type=str, choices=['uniform', 'lognormal'], required=True, help='Distribution to use for generating the TSP instance.')
parser.add_argument('--mutation_strategy', type=str, choices=['swap', 'scramble', 'wouter'], required=True, help='Mutation strategy to use.')

# Parse arguments
args = parser.parse_args()

# Convert string representations of lists to actual lists
sizes = ast.literal_eval(args.sizes)
ranges = ast.literal_eval(args.ranges)
if args.continuation == "":
    continuations = []
else:
    continuations = [",".join(map(str, tup)) for tup in ast.literal_eval(args.continuation)]

# Custom decoder function to convert specific JSON values back to their original types
def custom_decoder(obj):
    """
    Custom decoder function that converts specific JSON values back to their original types.
    Converts:
    - 'Infinity' to np.inf
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value == "Infinity":
                obj[key] = np.inf
            # elif isinstance(value, list):
                # Convert lists back to arrays
                # obj[key] = np.array(value)
            elif isinstance(value, dict):
                obj[key] = custom_decoder(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if value == "Infinity":
                obj[i] = np.inf
            # elif isinstance(value, list):
                # obj[i] = np.array(value)
            elif isinstance(value, dict):
                obj[i] = custom_decoder(value)
    return obj

def load_result(file_path):

  # Loading the JSON file with custom decoding
  with open(file_path, "r") as json_file:
      loaded_results = json.load(json_file, object_hook=custom_decoder)

  return loaded_results

# Modified Wouter's save function to give names to entries
def save_partial(configuration, results, citysize, rang, time, contin, is_final=False):
    """
    Saves the partial results of the experiment to a JSON file.
    """
    def custom_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.inf):
            return "np.inf"
        else:
            raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

    # Determine folder based on whether it's the final save
    if is_final:
        folder = f"Results/{args.distribution}_{args.tsp_type}"
    else:
        folder = f"Continuation/{args.distribution}_{args.tsp_type}" if contin else f"Results/{args.distribution}_{args.tsp_type}"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{folder}/result{citysize}_{rang}_{args.mutation_strategy}.json"
    
    # load the existing results
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            existing_results = json.load(file)
            # add the time to previous time
            existing_results["time"] += time
            # append the new results to the previous run
            existing_results["results"].update(results)
    else: # if the file does not exist, create an empty dictionary
        existing_results = {
            "time": time,
            "configuration": configuration,
            "results": results
        }
        
    # Save the results to a JSON file
    with open(filename, "w") as json_file:
        json.dump(existing_results, json_file, default=custom_encoder)
    
    # If final save, remove any continuation files
    if is_final:
        continuation_file = os.path.join("Continuation", f"{args.distribution}_{args.tsp_type}", f"result{citysize}_{rang}_{args.mutation_strategy}.json")
        if os.path.exists(continuation_file):
            os.remove(continuation_file)
                    

def initialize_hardest_matrix(citysize, rang, distribution, generation_type, mutation_type, continuations):
    """Load or generate the initial matrix and hardest value."""
    if f"{citysize},{rang}" in continuations:
        try:
            file_path = f"Continuation/{distribution}_{generation_type}/result{citysize}_{rang}_{mutation_type}.json"
            hardest, matrix = load_result(file_path)
            matrix = np.array(matrix)
        except Exception as e:
            ic("matrix not loaded, generating new TSP instance" + citysize + "_" + rang) 
            #print(f"{e}\n {citysize}_{rang} matrix not loaded, generating new TSP instance", flush=True)
            matrix = generate_tsp(citysize, generation_type, distribution, rang)
            hardest = 0
    else:
        matrix = generate_tsp(citysize, generation_type, distribution, rang)
        hardest = 0
    return hardest, matrix

def run_litals_algorithm(matrix):
    """Run Lital's algorithm and return results with error handling."""
    try:
        iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
        return iterations, optimal_tour, optimal_cost, None
    except Exception as e:
        return None, None, None, e

def process_mutation_iteration(j, matrix, hardest, hardest_matrix, mutation_type, generation_type, rang, distribution):
    """Process a single mutation iteration and return updated state."""
    iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
    
    if error:
        ic("Error in iteration {j}:", error)
        return hardest, hardest_matrix, matrix, None  # No results to record
    
    iteration_result = {
        "iterations": iterations,
        "hardest": hardest,
        "optimal_tour": optimal_tour,
        "optimal_cost": optimal_cost,
        "matrix": matrix.tolist(),
        "is_hardest": False
    }
    
    # Update hardest matrix if needed
    if iterations > hardest:
        hardest = iterations
        hardest_matrix = matrix.copy()
        iteration_result["is_hardest"] = True
    
    # Apply mutation to the hardest matrix
    new_matrix = apply_mutation(hardest_matrix, mutation_type, generation_type, rang, distribution)
    
    return hardest, hardest_matrix, new_matrix, iteration_result

def handle_saving(configuration, results, citysize, rang, start_time, continuations, should_save, is_final=False):
    if should_save:
        elapsed_time = time.time() - start_time
        save_partial(
            configuration, results, citysize, rang,
            elapsed_time, contin=(not is_final and f"{citysize},{rang}" in continuations),
            is_final=is_final
        )
        return {}
    return results

# Main Experiment Flow
def run_single_experiment(configuration, citysize, rang, mutations, continuations):
    """Run full experiment for a single citysize-range combination."""
    start_time = time.time()
    range_results = {}
    
    # Initialize matrix and hardest values
    hardest, matrix = initialize_hardest_matrix(
        citysize, rang,
        configuration["distribution"],
        configuration["generation_type"],
        configuration["mutation_type"],
        continuations
    )
    
    # Save initial matrix if not a continuation
    if f"{citysize},{rang}" not in continuations:
        range_results['initial_matrix'] = {
            "iterations": 0,
            "hardest": hardest,
            "optimal_tour": None,
            "optimal_cost": None,
            "matrix": matrix.tolist(),
            "is_hardest": True
        }
        save_partial(configuration, range_results, citysize, rang, 0, False)
        range_results = {}

    hardest_matrix = matrix.copy()
    
    # Mutation loop
    non_improved_iterations = 0
    for j in range(mutations):
        prev_hardest = hardest
        hardest, hardest_matrix, matrix, iteration_result = process_mutation_iteration(
            j, matrix, hardest, hardest_matrix,
            configuration["mutation_type"],
            configuration["generation_type"],
            rang,
            configuration["distribution"]
        )

        if hardest > prev_hardest:
            non_improved_iterations = 0
        else:
            non_improved_iterations += 1

        if non_improved_iterations >= 10000:
            ic(f"Stopping early after {j} iterations without improvement")
            break
        
        if iteration_result:
            range_results[f'iteration_{j}'] = iteration_result
            
        # Save every 100 iterations or when new hardest found
        save_condition = (j % 100 == 0) or (iteration_result and iteration_result["is_hardest"])
        range_results = handle_saving(
            configuration,
            range_results,
            citysize,
            rang,
            start_time,
            continuations,
            save_condition
        )
    
    # Final save for remaining results
    handle_saving(
        configuration, range_results, citysize, rang, 
        start_time, continuations, True, is_final=True
    )
    
    ic(f"Completed {mutations} mutations for citysize={citysize}, range={rang}")

# Refactored Experiment Function
def experiment(_cities, _ranges, _mutations, _continuations, generation_type, distribution, mutation_type):
    """Orchestrate experiments for all parameter combinations."""
    run_time = time.time()
    
    configuration = {
        "mutation_type": mutation_type,
        "generation_type": generation_type,
        "distribution": distribution,
    }
    
    for citysize in _cities:
        for rang in _ranges:
            config = configuration | {"city_size": citysize, "range": rang}
            run_single_experiment(
                config,
                citysize,
                rang,
                _mutations,
                _continuations
            )
    
    ic(f"Total duration: {time.time()-run_time:.2f}s")

# Run the experiment with args provided
experiment(sizes, ranges, args.mutations, continuations, args.tsp_type, args.distribution, args.mutation_strategy) 



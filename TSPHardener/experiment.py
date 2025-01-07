from algorithm import get_minimal_route
from generate_tsp import generate_tsp
from mutate_tsp import apply_mutation
import numpy as np
import json
import os
import glob
import time
# from icecream import ic

# Possible optimizations:
# import itertools
# import bisect
# # For the bash scripting
import argparse
import ast 

# Added some extra arguments to Wouter's experiment
# Initialize the parser
parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

# Add arguments
parser.add_argument('sizes', type=str, help='A list of city sizes, e.g., "[10,12]"')
parser.add_argument('ranges', type=str, help='A list of value ranges, e.g., "[10,1000]"') # control parameter
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
def save_partial(configuration, results, citysize, range, time, contin):
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
            # This will raise a TypeError for unknown types
            raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

    # Create the folder if it does not exist
    folder = f"Results/{args.distribution}_{args.tsp_type}" if not contin else f"Continuation/{args.distribution}_{args.tsp_type}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{folder}/result{citysize}_{range}_{args.mutation_strategy}.json"
    
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
    
    print(f"Partial results saved to {filename}", flush=True)
                    

'''
Experiment is defined as follows:
flowchart TD
    A[Start] --> B[Generate a Eucledian or Asymmetric TSP instance from uniform or lognormal distribution]
    B --> C[Run Lital’s Algorithm 
    hardness = Lital's iterations, 
    Output: matrix]
    C --> D[Evaluate Hardness:  hardness ≥ hardest?]
    D -- Yes --> E[Update hardest = hardness]
    E --> I[Save matrix hardness, hardest, repetitions, and duration]
    D -- No --> F[**Stopping Criteria**:
     Run more than 4000 repetitions or 5 days duration?]
    I --> F
    F -- Yes --> G[Stop and save data]
    F -- No --> H[Mutate the current hardest matrix with choosen mutation strategy]
    H --> C
'''

def experiment(_cities, _ranges, _mutations, _continuations, generation_type, distribution, mutation_type):
    """
    Runs the experiment with the specified parameters.

    Parameters:
    ----------
    _cities : int
        number of cities in the TSP instance.
    _ranges : int or float
        The control parameter for the distribution.
        - For the uniform , this is the upper bound for cost values in the matrix.
        - For the lognormal, this is the sigma parameter.
    _mutations : int
        The number of mutations to perform.
    _continuations : list of tuples
        Tuple of city size and range for which to continue the experiment.
    generation_type : str
        The type of TSP instance to generate (symmetric or asymmetric).
    distribution : str
        The distribution to use for generating the TSP instance (uniform or lognormal).
    mutation_type : str
        The mutation strategy to apply (swap, scramble, or Wouter's mutation).
    """
    run_time = time.time()
    for citysize in _cities:
        for rang in _ranges:
            configuration = { 
                "city_size": citysize,
                "range": rang,
                "mutation_type": mutation_type,
                "generation_type": generation_type,
                "distribution": distribution,
            }
            range_results = {}

            # Record the start time
            start_time = time.time()

            # Generate or load the matrix
            if f"{citysize},{rang}" in _continuations:
                try:
                    hardest, matrix = load_result(f"Continuation/{distribution}_{generation_type}/continue{citysize}_{rang}_{mutation_type}.json")
                    matrix = np.array(matrix)
                except Exception as e:
                    print(f"{e}\n {citysize}_{rang} matrix not loaded for {distribution}_{generation_type}_{mutation_type}, generating a TSP instance", flush=True)
                    matrix = generate_tsp(citysize, generation_type, distribution, rang)
                    hardest = 0
            else:
                matrix = generate_tsp(citysize, generation_type, distribution, rang)
                hardest = 0
                # Save the initial matrix
                range_results['initial_matrix'] = {
                    "iterations": 0,
                    "hardest": hardest,
                    "optimal_tour": None,
                    "optimal_cost": None,
                    "matrix": matrix.tolist(),
                    "is_hardest": True  # Mark initial matrix as hardest
                }
                # Save to json file
                save_partial(configuration, range_results, citysize, rang, 0, f"{citysize},{rang}" in _continuations)
                range_results = {}

            hardest_matrix = matrix

            for j in range(_mutations):
                try:
                    # Attempt to run Lital's algorithm
                    iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)

                    # Track the current iteration's results
                    range_results[f'iteration_{j}'] = {
                        "iterations": iterations,
                        "hardest": hardest,
                        "optimal_tour": optimal_tour,
                        "optimal_cost": optimal_cost,
                        "matrix": matrix.tolist(),
                        "is_hardest": False  # Default value
                    }

                    # Update if this matrix is the hardest found so far
                    if iterations > hardest:
                        hardest = iterations
                        hardest_matrix = matrix
                        range_results[f'iteration_{j}']["is_hardest"] = True

                    matrix = apply_mutation(matrix, mutation_type, generation_type, rang, distribution)

                except IndexError as e:
                    print(f"IndexError: {e} occurred during iteration {j}, skipping this mutation.", flush=True)
                    continue
                except RecursionError:
                    print(f"RecursionError occurred during iteration {j} with matrix shape {matrix.shape}. Retrying with another mutation.", flush=True)
                    matrix = apply_mutation(matrix, mutation_type, generation_type, rang, distribution)
                    continue
                except Exception as e:
                    print(f"Unexpected error: {e} during iteration {j}. Skipping this mutation.", flush=True)
                    continue

                # Save incrementally every 100 iterations or when a new hardest matrix is found
                if j % 100 == 0 or range_results[f'iteration_{j}']["is_hardest"]:
                    elapsed_time = time.time() - start_time
                    save_partial(configuration, range_results, citysize, rang, elapsed_time, f"{citysize},{rang}" in _continuations)
                    range_results = {}  # Clear the results to save incrementally

            # Final save for any remaining iteration results
            if range_results:
                elapsed_time = time.time() - start_time
                save_partial(configuration, range_results, citysize, rang, elapsed_time, f"{citysize},{rang}" in _continuations)

    elapsed_time = time.time() - run_time
    print(f"Done with cities = {citysize}, randMax = {rang}\nElapsed Time: {elapsed_time:.2f} seconds", flush=True)


# Run the experiment with args provided
experiment(sizes, ranges, args.mutations, continuations, args.tsp_type, args.distribution, args.mutation_strategy) 



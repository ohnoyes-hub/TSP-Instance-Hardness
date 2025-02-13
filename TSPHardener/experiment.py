from algorithm import get_minimal_route
from generate_tsp import generate_tsp
from mutate_tsp import apply_mutation
import numpy as np
import json
import os
import glob
import time
from icecream import ic # debugging
# job script
import argparse # for parsing arguments
import ast # for converting string to list

# Added some extra arguments to Wouter's experiment
# Initialize the parser
parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

# Add arguments
parser.add_argument('sizes', type=str, 
                    help='A list of city sizes, e.g., "[10,12]"')
parser.add_argument('ranges', type=str, 
                    help='A list of value ranges, e.g., "[10,1000]"')
parser.add_argument('mutations', type=int, 
                    help='An integer number of mutations, e.g., 500')
parser.add_argument('continuation', type=str, default="", nargs='?', 
                    help='A list of matrix continuations, e.g., "[(7,10),(50,10)]".')
parser.add_argument('--tsp_type', type=str, choices=['euclidean', 'asymmetric'], required=True,
                    help='Type of TSP to generate: symmetric or asymmetric.')
parser.add_argument('--distribution', type=str, choices=['uniform', 'lognormal'], required=True,
                    help='Distribution to use for generating the TSP instance.')
parser.add_argument('--mutation_strategy', type=str, choices=['swap', 'scramble', 'wouter'], required=True,
                    help='Mutation strategy to use.')

# Parse arguments
args = parser.parse_args()

# Convert string representations of lists to actual lists
sizes = ast.literal_eval(args.sizes)
ranges = ast.literal_eval(args.ranges)

# detect continuations based on existing files in Result or Continuation folders
continuations = []
for citysize in sizes:
    for rang in ranges:
        results_file = os.path.join("Results", f"{args.distribution}_{args.tsp_type}", 
                                    f"result{citysize}_{rang}_{args.mutation_strategy}.json")
        if os.path.exists(results_file):
            continue  # Skip completed experiments
        continuation_file = os.path.join("Continuation", f"{args.distribution}_{args.tsp_type}", f"result{citysize}_{rang}_{args.mutation_strategy}.json")
        if os.path.exists(continuation_file):
            continuations.append(f"{citysize},{rang}") # partial results exist for this configuration

# manual continuation: merge args.continuation with detected continuations
if args.continuation != "":
    # Parse manual continuations (e.g., "[(7,10),(50,10)]")
    manual_continuations = [f"{tup[0]},{tup[1]}" for tup in ast.literal_eval(args.continuation)]
    # Merge and deduplicate
    continuations = list(set(continuations + manual_continuations))

################################
# JSON (De)Serialization Helpers
################################
def custom_decoder(obj):
    """
    Custom decoder that converts "Infinity" to np.inf in nested structures.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value == "Infinity":
                obj[key] = np.inf
            elif isinstance(value, dict):
                obj[key] = custom_decoder(value)
            elif isinstance(value, list):
                obj[key] = custom_decoder(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if value == "Infinity":
                obj[i] = np.inf
            elif isinstance(value, dict):
                obj[i] = custom_decoder(value)
            elif isinstance(value, list):
                obj[i] = custom_decoder(value)
    return obj

def custom_encoder(obj):
    """
    Custom encoder that turns np types into standard JSON-friendly formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif obj is np.inf:
        return "Infinity"
    return obj

def load_result(file_path):
    """
    Loads the JSON file (partial results) from disk and returns
    (hardest, matrix) from the last iteration logged.
    
    The file structure has:
      {
        "time": <accumulated_time>,
        "configuration": {...},
        "results": {
            "iteration_0": {...},
            "iteration_1": {...},
            ...
        }
      }
    We find the iteration with the largest index.  
    """
    with open(file_path, "r") as json_file:
        loaded = json.load(json_file, object_hook=custom_decoder)

    # If there's an 'initial_matrix' key, that is iteration 0
    # Otherwise, we find the highest iteration_N in results
    all_iters = []
    for k,v in loaded["results"].items():
        if k.startswith("iteration_"):
            idx = int(k.split("_")[1])
            all_iters.append((idx, v))
        elif k == "initial_matrix":
            # Treat this as iteration -1 for ordering
            all_iters.append((-1, v))
    all_iters.sort(key=lambda x: x[0])  # sort by iteration index
    
    # The last logged iteration data
    _, last_data = all_iters[-1]
    
    # That last_data is something like:
    # {
    #     "iterations": <best so far>,
    #     "hardest": <integer>,
    #     "matrix": <2D list>,
    #     ...
    # }
    hardest = last_data["hardest"]
    matrix = np.array(last_data["matrix"])  # convert from list to np array
    
    return hardest, matrix

def save_partial(configuration, results, citysize, rang, time_spent, is_final=False):
    """
    Saves partial (or final) results to JSON.
    - If not final => goes to Continuation folder.
    - If final => goes to Results folder, removes any old Continuation file.
    """
    base_name = f"result{citysize}_{rang}_{args.mutation_strategy}.json"

    if is_final:
        folder = f"Results/{args.distribution}_{args.tsp_type}"
    else:
        folder = f"Continuation/{args.distribution}_{args.tsp_type}"

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    full_path = os.path.join(folder, base_name)

    # If file already exists, load it, update, and re-save
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            existing_data = json.load(f, object_hook=custom_decoder)
        # accumulate time
        existing_data["time"] += time_spent
        # merge results
        existing_data["results"].update(results)
    else:
        # new structure
        existing_data = {
            "time": time_spent,
            "configuration": configuration,
            "results": results
        }

    # Write
    with open(full_path, "w") as f:
        json.dump(existing_data, f, indent=2, default=custom_encoder)

    ic(f"Saved partial results to {full_path}")

    # If final, remove from Continuation (if it existed)
    if is_final:
        cont_path = os.path.join("Continuation", f"{args.distribution}_{args.tsp_type}", base_name)
        if os.path.exists(cont_path):
            os.remove(cont_path)
                    
####################################
# Main experiment logic and helpers
####################################
def initialize_hardest_matrix(citysize, rang, distribution, generation_type, mutation_type):
    """
    If there's an existing partial run for this (citysize, rang),
    load the hardest and the last matrix. Otherwise, generate a new TSP.
    """
    cont_file = os.path.join("Continuation", f"{distribution}_{generation_type}",
                             f"result{citysize}_{rang}_{mutation_type}.json")
    if os.path.exists(cont_file):
        try:
            hardest, matrix = load_result(cont_file)
            return hardest, matrix
        except Exception as e:
            ic(f"Error loading {cont_file}:", e)
    # fallback if no partial or error
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

def process_mutation_iteration(j, matrix, hardest, hardest_matrix, 
                               mutation_type, generation_type, rang, distribution):
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
            elapsed_time,
            is_final=is_final
        )
        return {}
    return results

#######################
# Main Experiment Flow
#######################
def run_single_experiment(configuration, citysize, rang, mutations):
    """
    Handles one (citysize, range) combination.
    If a results file is in Results, we skip.
    Otherwise we check Continuation for partial progress.
    """
    start_time = time.time()
    partial_results = {}

    # Initialize
    hardest, matrix = initialize_hardest_matrix(
        citysize, rang,
        configuration["distribution"],
        configuration["generation_type"],
        configuration["mutation_type"]
    )
    hardest_matrix = matrix.copy()

    # If we just started (no partial), store the initial matrix as iteration 0
    cont_file = os.path.join("Continuation", f"{args.distribution}_{args.tsp_type}",
                             f"result{citysize}_{rang}_{args.mutation_strategy}.json")
    if not os.path.exists(cont_file):
        partial_results['initial_matrix'] = {
            "iterations": 0,
            "hardest": hardest,
            "optimal_tour": None,
            "optimal_cost": None,
            "matrix": matrix.tolist(),
            "is_hardest": True
        }
        # Save as a brand new file => not final, not continuation
        save_partial(configuration, partial_results, citysize, rang,
                     time_spent=0, is_final=False)
        partial_results = {}

    # Main mutation loop
    non_improved_iterations = 0
    for j in range(mutations):
        prev_hardest = hardest
        hardest, hardest_matrix, matrix, iteration_data = process_mutation_iteration(
            j, matrix, hardest, hardest_matrix,
            configuration["mutation_type"],
            configuration["generation_type"],
            rang,
            configuration["distribution"]
        )
        if iteration_data is None:
            # Error, skip storing any iteration result
            continue

        # Check if improved
        if hardest > prev_hardest:
            non_improved_iterations = 0
        else:
            non_improved_iterations += 1

        # Early stop if no improvement for 10000 consecutive iterations
        if non_improved_iterations >= 10000:
            ic(f"Stopping early after {j} consecutive non-improving mutations.")
            break

        # Save iteration data
        partial_results[f'iteration_{j}'] = iteration_data

        # Save every 100 iterations or when new hardest found
        if (j % 100 == 0) or iteration_data["is_hardest"]:
            elapsed = time.time() - start_time
            save_partial(configuration, partial_results, citysize, rang,
                         time_spent=elapsed,
                         is_final=False)
            partial_results = {}

    # Final save
    elapsed = time.time() - start_time
    if partial_results:
        save_partial(configuration, partial_results, citysize, rang,
                     time_spent=elapsed,
                     is_final=True)

    ic(f"Completed up to {mutations} mutations for citysize={citysize}, range={rang}.")

def experiment(_cities, _ranges, _mutations):
    """
    Orchestrates the entire experiment for all (citysize, rang) combos.
    Skips combos that are already completed in Results.
    """
    run_time = time.time()
    config = {
        "mutation_type": args.mutation_strategy,
        "generation_type": args.tsp_type,
        "distribution": args.distribution,
    }
    for citysize in _cities:
        for rang in _ranges:
            # Skip only if not in continuations
            result_file = os.path.join("Results", f"{args.distribution}_{args.tsp_type}", f"result{citysize}_{rang}_{args.mutation_strategy}.json")
            if os.path.exists(result_file) and f"{citysize},{rang}" not in continuations:
                ic(f"Skipping citysize={citysize}, range={rang} (already in Results).")
                continue

            # Run single experiment otherwise
            conf_with_params = {
                **config,
                "city_size": citysize,
                "range": rang
            }
            run_single_experiment(conf_with_params, citysize, rang, _mutations)

    ic(f"Total experiment duration: {time.time()-run_time:.2f}s")
# 
experiment(sizes, ranges, args.mutations)


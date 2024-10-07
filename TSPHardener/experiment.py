from algorithm import get_minimal_route
from generate_tsp import generate_asymmetric_tsp, generate_euclidean_tsp
from mutate_tsp import shuffle, mutate, swap
import numpy as np
import json
import os
import glob
import time

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
parser.add_argument('ranges', type=str, help='A list of value ranges, e.g., "[10,1000]"')
parser.add_argument('mutations', type=int, help='An integer number of mutations, e.g., 500')
parser.add_argument('continuation', type=str, default="", nargs='?', help='A list of matrix continuations, e.g., "[(7,10),(50,10)]". Corresponding matrices must be in "Progress" folder')
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


# if args.continuation == "":
#     continuations = []
# else:
#     continuations = [",".join(map(str, tup)) for tup in ast.literal_eval(args.continuation)] 


def custom_encoder(obj):
    """
    Custom JSON encoder function that converts non-serializable objects.
    Converts:
    - numpy arrays to lists
    - numpy int64 to int
    - numpy float64 to float
    """
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

# using Wouter's version of the save_partial function to match his plotting function
def save_partial(results, citysize, range, time, contin):
    base_file_path = f"results{citysize}_{range}"
    file_extension = ".json"
    file_index = 0
    file_path = f"{base_file_path}{file_extension}"

    # Continue mode adds a suffix '_c'
    if contin:
        file_path = f"{base_file_path}_c{file_extension}"

    # Function to generate new file path with incrementing index
    def get_new_file_path():
        nonlocal file_index
        while True:
            new_file_path = f"{base_file_path}_{file_index}{file_extension}"
            if contin:
                new_file_path = f"{base_file_path}_c_{file_index}{file_extension}"
            if not os.path.exists(new_file_path):
                break
            elif os.path.getsize(new_file_path) <= 50 * 1024 * 1024:
                break
            file_index += 1
        return new_file_path

    # Check if the file exists and its size
    if os.path.exists(file_path) and os.path.getsize(file_path) >= 50 * 1024 * 1024:
        file_path = get_new_file_path()  # Get a new file path if current is too large

    # Check if the file exists to append or create new data structure
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            try:
                existing_data = json.load(json_file)
            except json.decoder.JSONDecodeError:
                existing_data = []
        existing_data = {
            "time": time,
            "configurations": results
        }
        #existing_data.append((time, results))
        data_to_write = existing_data
    else:
        existing_data = {
            "time": time,
            "configurations": results
        }
        data_to_write = [(time, results)]

    # Write the data to the file with custom encoding
    with open(file_path, "w") as json_file:
        json.dump(data_to_write, json_file, default=custom_encoder)

''' My version of the save_result function
def save_result(original_matrix, number_of_mutations, hardest_matrix, hardest_iterations, iteration_count, tsp_type, time_elapsed, filename):
    """
    Save the results to a JSON file.
    
    Parameters:
    ----------
    original_matrix : np.ndarray
        The original distance matrix.
    number_of_mutations : int
        The number of mutations performed.
    hardest_matrix : np.ndarray
        The hardest found distance matrix.
    hardest_iterations : int
        The number of iterations needed to find the hardest matrix.
    iteration_count : np.ndarray
        The number of iterations for each mutation.
    tsp_type : str
        The type of the TSP instance.
    time_elapsed : float
        The time elapsed during the experiment.
    filename : str
        The name of the file to save the results.
    """
    # If not file exists, create a new one
    if not os.path.exists(filename):
        open(filename, "w").close()

    # If file exists, append to the existing one(we are continuing an experiment)

    data = {
        "original_matrix": original_matrix.tolist(),
        "number_of_mutations": number_of_mutations,
        "hardest_matrix": hardest_matrix.tolist(),
        "hardest_iterations": hardest_iterations,
        "hardness": iteration_count,
        "tsp_type": tsp_type,
        "time_elapsed": time_elapsed
    }

    # Save the data to a JSON file
    with open(filename, "w") as file:
        json.dump(data, file)
'''

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

def generate_tsp_instance(city_size, generation_type, distribution, upper_bound):
    """
    Generate a TSP instance with the specified parameters.
    
    Parameters:
    ----------
    city_size : int
        The number of cities in the TSP instance.
    generation_type : str
        The type of TSP instance to generate (symmetric or asymmetric).
    distribution : str
        The distribution to use for generating the TSP instance (uniform or lognormal).
    upper_bound : int
        The upper bound for cost values in the matrix.
        
    Returns:
    -------
    np.ndarray
        The generated TSP instance.
    """
    if generation_type == "euclidean":
        return generate_euclidean_tsp(city_size, distribution, upper_bound)
    elif generation_type == "asymmetric":
        return generate_asymmetric_tsp(city_size, distribution, upper_bound)
    else:
        raise ValueError("Invalid generation type. Choose either 'euclidean' or 'asymmetric'.")

def apply_mutation(matrix, mutation_type, tsp_type, upper):
    """
    Apply a mutation to the given TSP instance.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The TSP instance to mutate.
    mutation_type : str
        The mutation strategy to apply (swap, scramble, or Wouter's mutation).
    tsp_type : str
        The type of TSP instance (euclidean or asymmetric).
    upper : int
        The upper bound for cost values in the matrix.
        
    Returns:
    -------
    np.ndarray
        The mutated TSP instance.
    """
    if mutation_type == "swap":
        return swap(tsp_type, matrix)
    elif mutation_type == "scramble":
        return shuffle(tsp_type, matrix)
    elif mutation_type == "wouter":
        return mutate(tsp_type, matrix, upper)
    else:
        raise ValueError("Invalid mutation type. Choose either 'swap', 'scramble', or 'wouter'.")

def experiment(_cities, _ranges, _mutations, _continuations, generation_type, distribution, mutation_type):
    for citysize in _cities:
        for rang in _ranges:
            range_results = {
                "city_size": citysize,
                "range": rang,
                "mutation_type": mutation_type,
                "generation_type": generation_type,
                "distribution": distribution,
            }

            # Record the start time
            start_time = time.time()

            # Generate or load the matrix
            if f"{citysize},{rang}" in _continuations:
                try:
                    hardest, matrix = load_result(f"Progress/continue{citysize}_{rang}.json")
                    matrix = np.array(matrix)
                except Exception as e:
                    print(f"{e}\n {citysize}_{rang} matrix not loaded, generating random matrix", flush=True)
                    matrix = generate_tsp_instance(citysize, generation_type, distribution, rang)
                    hardest = 0
            else:
                matrix = generate_tsp_instance(citysize, generation_type, distribution, rang)
                hardest = 0

            for j in range(_mutations):
                try:
                    # Run Lital's algorithm on swap and scramble mutation has a chance to throw RecursionError
                    iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
                except RecursionError:
                    save_partial(range_results, citysize, rang, time.time() - start_time, f"{citysize},{rang}" in _continuations)
                    print(f"RecursionError occurred on matrix {j} with shape {matrix.shape}. Retrying with another mutation", flush=True)
                    matrix = apply_mutation(matrix, mutation_type, generation_type, rang)
                    continue

                #range_results[j] = (iterations, hardest, optimal_tour, optimal_cost, matrix)
                range_results[j] = {
                    "iterations": iterations,
                    "hardest": hardest,
                    "optimal_tour": optimal_tour,
                    "optimal_cost": optimal_cost,
                    "matrix": matrix.tolist()
                }

                # Apply the selected mutation strategy
                if iterations >= hardest:
                    hardest_matrix = matrix
                    matrix = apply_mutation(hardest_matrix, mutation_type, generation_type, rang)
                    hardest = iterations
                else:
                    matrix = apply_mutation(hardest_matrix, mutation_type, generation_type, rang)

                if j > 0 and (j + 1) % 100 == 0:
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    # Save to json file
                    save_partial(range_results, citysize, rang, elapsed_time, f"{citysize},{rang}" in _continuations)
                    range_results = {}

    elapsed_time = time.time() - start_time
    print(f"Done with cities = {citysize}, randMax = {rang}\nElapsed Time: {elapsed_time:.2f} seconds", flush=True)




experiment(sizes, ranges, args.mutations, continuations, args.tsp_type, args.distribution, args.mutation_strategy)
# if __name__ == "__main__":
#     experiment(sizes, ranges, args.mutations, continuations, args.tsp_type, args.distribution, args.mutation_strategy)
    
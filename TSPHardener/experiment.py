from algorithm import get_minimal_route
from generate_tsp import generate_asymmetric_tsp, generate_euclidean_tsp
from mutate_tsp import permute_matrix, permute_symmetric_matrix, swap_mutate, mutate_matrix, mutate_matrix_symmetric
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

# Copied from Woulter's experiment.py
# Initialize the parser
parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

# Add arguments
# Add arguments
parser.add_argument('--variant', type=str, default="atsp", help='atsp if the TSP is asymmetric, etsp if Euclidean.')
parser.add_argument('--is_uniform', type=str, default="true", help='True if the matrix is generated with uniform distribution, False if lognormal.')
parser.add_argument('--sizes', type=str, default="[15,20]", help='A list of city sizes, e.g., "[10,12]"')
parser.add_argument('--control', type=str, default="[10, 50]", help='A list of control parameter ranges, e.g., "[10,1000]"')
parser.add_argument('--mutation_strategy', type=str, default="swap", help='The mutation strategy to use, "inplace", "swap", "shuffle"')
parser.add_argument('--mutations', type=int, default=300, help='An integer number of mutations, e.g., 500')
parser.add_argument('--continue_', type=str, default="false", help='True if the experiment should continue from a partial experiment.')

args = parser.parse_args()

# Parsing the arguments to the appropriate types
variant = args.variant
is_uniform = args.is_uniform.lower() == "true"
sizes = ast.literal_eval(args.sizes)
control = ast.literal_eval(args.control)
mutation_strategy = args.mutation_strategy
mutations = args.mutations
continue_ = args.continue_.lower() == "true"

print("Variant:", variant)
print("Is Uniform:", is_uniform)
print("Sizes:", sizes)
print("Control:", control)
print("Mutation Strategy:", mutation_strategy)
print("Mutations:", mutations)
print("Continue:", continue_)

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
# Stopped copying here


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
def experiment(tsp_variant, is_uniform, sizes, control, mutation_strategy, mutations, continue_):
    """
    Defines the experiment setup for creating harder TSP instances.

    Parameters:
    ----------
    tsp_variant : str
        atsp if the TSP is asymmetric, etsp if Euclidean.
    is_uniform : bool
        True if the matrix is generated with uniform distribution, False if lognormal.
    sizes : list
        A list of city sizes.
    control : list
        A list of control parameter ranges.
    mutation_strategy : str
        The mutation strategy to use, "inplace", "swap", "shuffle".
    mutations : int
        The number of mutations to perform.
    continue_
        True if the experiment should continue from a previous matrix. 
    """

    iteration_counter = []

    for size in sizes:
        for range in control:
            if tsp_variant == "atsp":    
                matrix = generate_asymmetric_tsp(n=size, isUniform=is_uniform, dimensions=size)
            elif tsp_variant == "etsp":
                matrix = generate_euclidean_tsp(n=size, isUniform=is_uniform, dimensions=size)
            else:
                raise ValueError("Invalid TSP variant. Choose 'atsp' or 'etsp'.")   

            # Start timer
            start_time = time.time()

            # Initialize the numer of solutions found:    
            iterations, _, _ = get_minimal_route(matrix) # optimal_tour, optimal_cost are not used
            hardest = 0
            for mutation in range(mutations):
                # Find the minimal route with Little's algorithm
                iterations, _, _ = get_minimal_route(matrix)
                iteration_counter.append(iterations)

                # New hardest instance found or no difference found in mutation
                if iterations >= hardest: 
                    hardest_matrix = matrix
                    hardest = iterations
                    # print("New hardest instance found with", iterations, "Lital's recursions.")
                    # Do the appropriate mutation for the TSP type
                    if mutation_strategy == "swap" and tsp_variant == "atsp":
                        matrix = swap_mutate(hardest_matrix)
                    elif mutation_strategy == "inplace" and tsp_variant == "atsp":
                        matrix = mutate_matrix(hardest_matrix, range)
                    elif mutation_strategy == "shuffle" and tsp_variant == "atsp":
                        matrix = permute_matrix(hardest_matrix)
                    elif mutation_strategy == "shuffle" and tsp_variant == "etsp":
                        matrix = permute_symmetric_matrix(hardest_matrix)
                    elif mutation_strategy == "inplace" and tsp_variant == "etsp":
                        matrix = mutate_matrix_symmetric(hardest_matrix, range)
                    #elif mutation_strategy == "swap" and tsp_variant == "etsp":
                        #matrix = swap_mutate_symmetric(hardest_matrix, False) # TODO: implement swap_mutate_symmetric
                    else:
                        raise ValueError("Invalid mutation strategy. Choose 'inplace', 'swap', or 'shuffle'.")
                else: # Try another permutation
                    if mutation_strategy == "swap" and tsp_variant == "atsp":
                        matrix = swap_mutate(hardest_matrix)
                    elif mutation_strategy == "inplace" and tsp_variant == "atsp":
                        matrix = mutate_matrix(hardest_matrix, range)
                    elif mutation_strategy == "shuffle" and tsp_variant == "atsp":
                        matrix = permute_matrix(hardest_matrix)
                    elif mutation_strategy == "shuffle" and tsp_variant == "etsp":
                        matrix = permute_symmetric_matrix(hardest_matrix)
                    elif mutation_strategy == "inplace" and tsp_variant == "etsp":
                        matrix = mutate_matrix_symmetric(hardest_matrix, range)
                    # elif mutation_strategy == "swap" and tsp_variant == "etsp":
                        # matrix = swap_mutate_symmetric(hardest_matrix, False) # TODO: implement swap_mutate_symmetric
                    else:
                        raise ValueError("Invalid mutation strategy. Choose 'inplace', 'swap', or 'shuffle'.")
          
                if mutation > 0 and mutation % 100 == 0:
                    print(f"Mutation {mutation} completed.")
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    # save to json file
                    if elapsed_time > 5*24*60*60 or mutation > 4000:
                        filename = f"results_{size}_{range}.json"
                        save_result(matrix, mutations, hardest_matrix, hardest, iteration_counter, tsp_variant, elapsed_time, filename)
                        break # Stop the experiment
                    else:
                        # experiment did not finish, give filename with continue_ flag
                        filename = f"results_{size}_{range}_continue.json"
                        save_result(matrix, mutations, hardest_matrix, hardest, iteration_counter, tsp_variant, elapsed_time, filename)
                    # save_result(matrix, mutations, hardest_matrix, hardest, iteration_counter, tsp_variant, elapsed_time, "")
                    
                        

# TODO: read the matrix from the JSON file
# def experiment(_is_atsp, upperbound_cost, mutations, _continue):
#     """
#     Defines the experiment setup for creating harder TSP instances.

#     Parameters:
#     ----------
#     _is_atsp : bool
#         True if the TSP is asymmetric, False if Euclidean.
#     upperbound_costs : int
#         The upper bound for cost values in the matrix.
#     mutations : int
#         The number of mutations to perform.
#     _continue : bool
#         True if the experiment should continue from a previous matrix.
#     """
#     hardest = 0
#     hardness_counter = []
#     optimal_cost_counter = []
#     iteration_counter = []
#     if _is_atsp:
#         matrix = generate_asymmetric_tsp(n=20, upper=upperbound_cost)
#     else:
#         matrix = generate_symmetric_matrix(n=20, upper=upperbound_cost)

#     # Start timer
#     start_time = time.time()

#     # Initialize the numer of solutions found:    
#     iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)

#     for mutation in range(mutations):
#         # Find the minimal route with Little's algorithm
#         iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
#         iteration_counter.append(iterations)
#         # New hardest instance found or no difference found in mutation
#         if iterations >= hardest: 
#             hardest_matrix = matrix
#             print("New hardest instance found with", iterations, "Lital's recursions.")
#             # Do the appropriate muation for the TSP type
#             if _is_atsp:
#                 #matrix = swap_mutate(hardest_matrix, False) # swap mutate generally creates more instances with higher iterations.
#                 matrix = permute_matrix(hardest_matrix, False) # permute matrix creates harder instances but with less instances.
#             else:
#                 matrix = permute_symmetric_matrix(hardest_matrix, False)
#                 # matrix = swap_mutate_symmetric(matrix, False)
#             hardest = iterations
#             # Save to plot the results
#             hardness_counter.append(hardest)
#             optimal_cost_counter.append(optimal_cost)
#         else: # Try another permutation
#             matrix = permute_matrix(hardest_matrix, False)
    
#     # End timer
#     end_time = time.time()

#     # It sometimes prints even if Little's algorithm is still running? Might be some python parallelism thing?
#     print("Tracking hardness:", hardness_counter) 
#     print("Optimal cost:", optimal_cost_counter)
#     print("Iterations:", iteration_counter)
#     print("Time elapsed:", end_time - start_time)

#     # Save the results to a JSON file
#     filename = f"initial_results_save.json"

#     save_result(matrix, mutations, hardest_matrix, hardest, optimal_cost_counter, iteration_counter, "ATSP" if _is_atsp else "TSP", end_time - start_time, filename)

if __name__ == "__main__":
    matrix = generate_euclidean_tsp(10, True, 15)
    mutate_matrix_symmetric(matrix, True)

    
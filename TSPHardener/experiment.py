from algorithm import get_minimal_route
from generate_tsp import generate_asymmetric_tsp, generate_symmetric_matrix
from mutate_tsp import permute_matrix, permute_symmetric_matrix, swap_mutate, swap_mutate_symmetric
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
# parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

# # Add arguments
# parser.add_argument('sizes', type=str, help='A list of city sizes, e.g., "[10,12]"')
# parser.add_argument('ranges', type=str, help='A list of value ranges, e.g., "[10,1000]"')
# parser.add_argument('mutations', type=int, help='An integer mutations, e.g., 500')
# parser.add_argument('continuation', type=str, default="", nargs='?', help='A list of matrix continuations, e.g., "[(7,10),(50,10)]". Corresponding matrices must be in "Progress" folder')

# # Parse arguments
# args = parser.parse_args()

# # Convert string representations of lists to actual lists
# sizes = ast.literal_eval(args.sizes)
# ranges = ast.literal_eval(args.ranges)
# if args.continuation == "":
#     continuations = []
# else:
#     continuations = [",".join(map(str, tup)) for tup in ast.literal_eval(args.continuation)] 
# Stopped copying here


def save_result(original_matrix, number_of_mutations, hardest_matrix, hardest_iterations, optimal_cost, iteration_count, tsp_type, time_elapsed, filename):
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
    optimal_cost : np.ndarray
        The optimal costs of each hardest matrix.
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

    # TODO: ask if I should track any other data.
    data = {
        "original_matrix": original_matrix.tolist(),
        "number_of_mutations": number_of_mutations,
        "hardest_matrix": hardest_matrix.tolist(),
        "hardest_iterations": hardest_iterations,
        "optimal_cost": optimal_cost,
        "iteration_count": iteration_count,
        "tsp_type": tsp_type
    }

    # Save the data to a JSON file
    with open(filename, "w") as file:
        json.dump(data, file)

# TODO: read the matrix from the JSON file
# TODO: add partial results
def experiment(_is_atsp, upperbound_cost, mutations, _continue):
    """
    Defines the experiment setup for creating harder TSP instances.

    Parameters:
    ----------
    _is_atsp : bool
        True if the TSP is asymmetric, False if Euclidean.
    upperbound_costs : int
        The upper bound for cost values in the matrix.
    mutations : int
        The number of mutations to perform.
    _continue : bool
        True if the experiment should continue from a previous matrix.
    """
    hardest = 0
    hardness_counter = []
    optimal_cost_counter = []
    iteration_counter = []
    if _is_atsp:
        matrix = generate_asymmetric_tsp(n=20, upper=upperbound_cost)
    else:
        matrix = generate_symmetric_matrix(n=20, upper=upperbound_cost)

    # Start timer
    start_time = time.time()

    # Initialize the numer of solutions found:    
    iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)

    for mutation in range(mutations):
        # Find the minimal route with Little's algorithm
        iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
        iteration_counter.append(iterations)
        # New hardest instance found or no difference found in mutation
        if iterations >= hardest: 
            hardest_matrix = matrix
            print("New hardest instance found with", iterations, "Lital's recursions.")
            # Do the appropriate muation for the TSP type
            if _is_atsp:
                #matrix = swap_mutate(hardest_matrix, False) # swap mutate generally creates more instances with higher iterations.
                matrix = permute_matrix(hardest_matrix, False) # permute matrix creates harder instances but with less instances.
            else:
                matrix = permute_symmetric_matrix(hardest_matrix, False)
                # matrix = swap_mutate_symmetric(matrix, False)
            hardest = iterations
            # Save to plot the results
            hardness_counter.append(hardest)
            optimal_cost_counter.append(optimal_cost)
        else: # Try another permutation
            matrix = permute_matrix(hardest_matrix, False)
    
    # End timer
    end_time = time.time()

    # It sometimes prints even if Little's algorithm is still running? Might be some python parallelism thing?
    print("Tracking hardness:", hardness_counter) 
    print("Optimal cost:", optimal_cost_counter)
    print("Iterations:", iteration_counter)
    print("Time elapsed:", end_time - start_time)

    # Save the results to a JSON file
    filename = f"initial_results_save.json"

    save_result(matrix, mutations, hardest_matrix, hardest, optimal_cost_counter, iteration_counter, "ATSP" if _is_atsp else "TSP", end_time - start_time, filename)

if __name__ == "__main__":
    # experiment(_is_atsp=False, upperbound_cost=100, mutations=100, _continue=False)
    
    # experiment(_is_atsp=True, upperbound_cost=100, mutations=100, _continue=False)
    # Result with permute_matrix of size 20
   
    # Testing with swap_mutate
    # experiment(_is_atsp=True, upperbound_cost=100, mutations=100, _continue=False)
    
    # Testing saving data
    #experiment(_is_atsp=True, upperbound_cost=100, mutations=20, _continue=False)

    #matrix = generate_asymmetric_tsp(n=4, upper=100)
    #permute_matrix(matrix, True)

    # matrix = generate_asymmetric_tsp(n = 4, upper=100)
    # matrix_mut = mutate_matrix(matrix, 100, True)
    # print(matrix.round(1))
    # print("...............")
    # print(matrix_mut.round(1))

    # matrix_swap = swap_mutate(matrix, True) 
    # matrix_perm = permute_matrix(matrix, True)

    experiment(_is_atsp=True, upperbound_cost=100, mutations=100, _continue=False)
from algorithm import get_minimal_route
from generate_tsp import generate_asymmetric_tsp, generate_symmetric_matrix
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



def permute_matrix(matrix, _print) -> np.ndarray:
    """
    Generate a random permutation of the given distance matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    _print : bool
        Print results in the console if True.
    """
    n = len(matrix)
    
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
    permuted_matrix = matrix.copy()
    
    # TODO: find a faster way to permute the matrix. This is essentially brute force O(n^2 - n) where we shuffle list of n^2 - n elements in n x n matrix.
    # List of indices excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Create a flattened list of values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]
    
    # Shuffle the values
    np.random.shuffle(values)
    
    # Assign the shuffled values back to the matrix
    for (i, j), value in zip(indices, values):
        permuted_matrix[i, j] = value
    
    if _print:
        print(f"Permuted matrix:\n{permuted_matrix.round(1)}")
    
    return permuted_matrix

def permute_symmetric_matrix(matrix: np.ndarray, _print) -> np.ndarray:
    """
    Generate a random permutation of the given symmetric cost matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    _print : bool
        Print results in the console if True.
    """
    n = len(matrix)
    
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
    permuted_matrix = matrix.copy()

    # list of indices in the upper triangle excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # upper triangle values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]

    # Shuffle the values to randomize their order
    np.random.shuffle(values)

    # Assign the shuffled values back to the upper triangle
    for (i, j), value in zip(indices, values):
        permuted_matrix[i, j] = value
        permuted_matrix[j, i] = value  # Mirror the value in the lower triangle
    
    if _print:
        print(f"Permuted matrix:\n{permuted_matrix.round(1)}")
    
    return permuted_matrix


# Less agressive mutation. Permute only two elements in the matrix.
def swap_mutate(matrix, _print) -> np.ndarray:
    """
    Swap two random elements in the cost matrix excluding the diagonal.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to mutate.
    _print : bool
        Print results in the console if True.
    """
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
    n = len(matrix)
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    
    while i == j:
        j = np.random.randint(0, n)
    
    matrix[i, j], matrix[j, i] = matrix[j, i], matrix[i, j]
    
    if _print:
        print(f"Mutated matrix:\n{matrix.round(1)}")
    
    return matrix

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
            print("New hardest instance found with", iterations, "iterations.")
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
    # print("Iterations:", iteration_counter)
    print("Time elapsed:", end_time - start_time)

    # Save the results to a JSON file
    filename = f"initial_results_save.json"

    save_result(matrix, mutations, hardest_matrix, hardest, optimal_cost_counter, iteration_counter, "ATSP" if _is_atsp else "TSP", end_time - start_time, filename)

if __name__ == "__main__":
    # experiment(_is_atsp=False, upperbound_cost=100, mutations=100, _continue=False)
    # Initial result with permute_symmetric_matrix size 20:
    # New hardest instance found with 78 iterations.
    # New hardest instance found with 452 iterations.
    # New hardest instance found with 1198 iterations.
    # Tracking hardness: [78, 452, 1198]
    # Optimal cost: [170.33748847353033, 201.79157683701226, 177.47932307772373]
    # Iterations: [78, 452, 1198, 388, 40, 56, 19, 291, 414, 339, 41, 122, 18, 53, 72, 325, 18, 155, 152, 18, 31, 62, 30, 100, 59, 78, 45, 93, 98, 126, 161, 102, 127, 24, 122, 100, 189, 65, 44, 22, 167, 464, 74, 82, 188, 128, 91, 58, 115, 64, 102, 117, 46, 22, 370, 31, 52, 150, 226, 29, 78, 34, 71, 106, 73, 71, 63, 136, 45, 226, 39, 74, 129, 70, 202, 77, 27, 64, 166, 69, 702, 81, 32, 82, 165, 63, 63, 80, 32, 26, 143, 219, 148, 18, 130, 316, 27, 124, 122, 55]

    # experiment(_is_atsp=True, upperbound_cost=100, mutations=100, _continue=False)
    # Result with permute_matrix of size 20
    # New hardest instance found with 73 iterations.
    # New hardest instance found with 248 iterations.
    # New hardest instance found with 252 iterations.
    # New hardest instance found with 265 iterations.
    # New hardest instance found with 270 iterations.
    # New hardest instance found with 565 iterations.
    # Tracking hardness: [73, 248, 252, 265, 270, 565]
    # Optimal cost: [171.95309681281128, 197.79455456237363, 173.10504927926712, 174.7876324629744, 184.44385266135941, 225.26006718485652]
    # Iterations: [73, 248, 110, 36, 198, 232, 33, 69, 137, 53, 85, 82, 61, 46, 26, 87, 177, 93, 194, 51, 34, 151, 153, 252, 265, 203, 124, 27, 190, 31, 270, 155, 109, 78, 32, 565, 43, 128, 124, 60, 160, 102, 50, 34, 79, 120, 84, 94, 70, 118, 121, 41, 90, 94, 54, 203, 170, 29, 184, 97, 44, 34, 57, 124, 37, 20, 54, 100, 58, 50, 138, 18, 40, 94, 197, 85, 80, 45, 170, 44, 18, 98, 63, 90, 404, 18, 221, 46, 30, 38, 141, 100, 42, 78, 54, 66, 270, 29, 133, 129]
    
    # Testing with swap_mutate
    # experiment(_is_atsp=True, upperbound_cost=100, mutations=100, _continue=False)
    # New hardest instance found with 52 iterations.
    # New hardest instance found with 79 iterations.
    # New hardest instance found with 89 iterations.
    # New hardest instance found with 89 iterations.
    # New hardest instance found with 89 iterations.
    # New hardest instance found with 89 iterations.
    # New hardest instance found with 120 iterations.
    # New hardest instance found with 120 iterations.
    # New hardest instance found with 120 iterations.
    # New hardest instance found with 120 iterations.
    # New hardest instance found with 120 iterations.
    # New hardest instance found with 347 iterations.
    # New hardest instance found with 634 iterations.
    # New hardest instance found with 634 iterations.
    # New hardest instance found with 634 iterations.
    # New hardest instance found with 661 iterations.
    # New hardest instance found with 661 iterations.
    # New hardest instance found with 1534 iterations.
    # New hardest instance found with 1534 iterations.
    # New hardest instance found with 1534 iterations.
    # New hardest instance found with 1534 iterations.
    # New hardest instance found with 1542 iterations.
    # Tracking hardness: [52, 79, 89, 89, 89, 89, 120, 120, 120, 120, 120, 347, 634, 634, 634, 661, 661, 1534, 1534, 1534, 1534, 1542]
    # Optimal cost: [173.878501387277, 180.39451453239656, 180.39451453239656, 180.39451453239656, 180.39451453239656, 180.39451453239656, 178.26629524116387, 178.26629524116387, 178.26629524116387, 178.26629524116387, 178.26629524116387, 212.43973730018917, 208.62603802557533, 208.62603802557533, 208.62603802557533, 208.62603802557533, 208.62603802557533, 211.56119110861295, 211.56119110861295, 211.56119110861295, 211.56119110861295, 211.56119110861295]
    # Iterations: [52, 46, 79, 89, 89, 89, 89, 120, 120, 120, 120, 120, 111, 347, 346, 58, 122, 634, 634, 634, 661, 661, 1534, 1534, 1534, 1534, 1542, 1521, 109, 47, 161, 64, 77, 133, 429, 23, 100, 51, 155, 184, 137, 69, 127, 135, 177, 50, 110, 48, 34, 475, 73, 77, 144, 72, 38, 49, 123, 230, 118, 76, 175, 92, 174, 69, 127, 37, 152, 257, 127, 18, 42, 63, 34, 52, 32, 28, 63, 51, 327, 72, 31, 41, 90, 437, 23, 57, 80, 52, 105, 100, 98, 97, 73, 36, 176, 47, 211, 127, 107, 42]

    # Testing saving data
    experiment(_is_atsp=True, upperbound_cost=100, mutations=20, _continue=False)
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
# import argparse
# import ast 

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


# def save_partial_results(matrix, filename):
    """
    Save the partial results of hardest found matrix to a JSON file.

    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to save.
    filename : str
        The name of the file to save the matrix.
    """


    
# Less agressive mutation. Permute only two elements in the matrix.
def mutate_elements_together(matrix, _print):
    """
    Mutates two elements with each other.

    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    _print : bool
        Print results in the console if True.
    """

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
    if _is_atsp:
        matrix = generate_asymmetric_tsp(n=10, upper=upperbound_cost)
        #print("Matrix:", matrix)
        #permute_matrix(matrix, True)
    else:
        matrix = generate_symmetric_matrix(n=10, upper=upperbound_cost)
        print("Matrix:", matrix)
    
    iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
    # Woulter only consider's interations, I will also investigate the optimal cost
    #print("Interations:", iterations, "Optimal cost:", optimal_cost)
    #print("Interations:", iterations, "Optimal cost:", optimal_cost)
    #iterations, optimal_tour, optimal_cost = get_minimal_route(permute_matrix(matrix, False))
    #print("Interations:", iterations, "Optimal cost:", optimal_cost)

    hardest = 0
    for mutation in range(mutations):
        # Find the minimal route with Little's algorithm
        iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)

        # New hardest instance found
        if iterations >= hardest: 
            hardest_matrix = matrix
            print("New hardest instance found with", iterations, "iterations.")
            matrix = permute_matrix(hardest_matrix, False)
            #print("New matrix:", matrix)
            hardest = iterations
        else: # Try another permutation
            matrix = permute_matrix(hardest_matrix, False)


if __name__ == "__main__":
    experiment(False, 100, 5, False)
    experiment(True, 100, 5, False)
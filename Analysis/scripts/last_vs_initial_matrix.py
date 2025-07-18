import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_experiment import load_json
from icecream import ic

def calculate_frobenius_difference(data):
    """
    Calculate the Frobenius norm between the iteration_0 matrix and last_matrix.
    """
    errors = []
    
    # Check if required structures exist
    if 'results' not in data:
        errors.append("Missing 'results' in data")
        return None, errors
    results = data['results']
    
    if 'hard_instances' not in results or 'last_matrix' not in results:
        errors.append("Missing 'hard_instances' or 'last_matrix' in results")
        return None, errors
    
    hard_instances = results['hard_instances']
    if 'iteration_0' not in hard_instances:
        errors.append("Missing 'iteration_0' in hard_instances")
        return None, errors
    
    iteration_0_matrix = hard_instances['iteration_0'].get('matrix')
    last_matrix = results['last_matrix']
    
    if iteration_0_matrix is None:
        errors.append("'iteration_0' does not contain a 'matrix'")
        return None, errors
    
    # Validate matrix structures
    if not (isinstance(iteration_0_matrix, list) or not all(isinstance(row, list) for row in iteration_0_matrix)):
        errors.append("'iteration_0' matrix is not a 2D list")
    if not (isinstance(last_matrix, list) or not all(isinstance(row, list) for row in last_matrix)):
        errors.append("'last_matrix' is not a 2D list")
    if errors:
        return None, errors
    
    # Check matrix dimensions
    rows_iter0 = len(iteration_0_matrix)
    cols_iter0 = len(iteration_0_matrix[0]) if rows_iter0 > 0 else 0
    rows_last = len(last_matrix)
    cols_last = len(last_matrix[0]) if rows_last > 0 else 0
    
    if rows_iter0 != rows_last or cols_iter0 != cols_last:
        errors.append("Matrices have different dimensions")
        return None, errors
    
    # Calculate Frobenius norm
    total = 0.0
    for i in range(rows_iter0):
        if len(iteration_0_matrix[i]) != cols_iter0 or len(last_matrix[i]) != cols_last:
            errors.append("Inconsistent row lengths in matrices")
            return None, errors
        for j in range(cols_iter0):
            try:
                a = float(iteration_0_matrix[i][j])
                b = float(last_matrix[i][j])
                total += (a - b) ** 2
            except (ValueError, TypeError):
                errors.append(f"Non-numeric element at position ({i}, {j})")
                return None, errors
    
    frobenius = total ** 0.5
    return frobenius, errors

data, load_errors, warnings = load_json("Results/uniform_euclidean/city20_range20_scramble.json")

if data is None:
    ic(load_errors)
else:
    # Calculate Frobenius norm
    frobenius, calc_errors = calculate_frobenius_difference(data)
    if calc_errors:
        ic(calc_errors)
    else:
        ic(frobenius)
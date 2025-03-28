import numpy as np
from matplotlib.lines import Line2D
import os
import json
from utils.json_utils import custom_decoder
from icecream import ic
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analysis_util.load_json import load_json

# approach 1
def compute_triangle_inequality_metrics(matrix):
    matrix = np.array(matrix)
    n = len(matrix)
    violations = []
    
    # Sample 1000 triplets to reduce computation (adjust as needed)
    for _ in range(1000):
        i, j, k = np.random.choice(n, 3, replace=False)
        direct = matrix[i, k]
        indirect = matrix[i, j] + matrix[j, k]
        if indirect == 0:
            continue  # Avoid division by zero
        ratio = direct / indirect
        if ratio > 1:
            violations.append(ratio)
    
    if not violations:
        return {
            "avg_violation": 0,
            "max_violation": 0,
            "violation_freq": 0
        }
    
    return {
        "avg_violation": np.mean(violations),
        "max_violation": np.max(violations),
        "violation_freq": len(violations) / 1000 
    }

# approach 2
def triangle_inequality_violation(matrix):
    matrix = np.array(matrix)
    n = matrix.shape[0]

    # Mask diagonals as they're infinite in TSP
    mask = ~np.eye(n, dtype=bool)

    violation_count = 0
    violation_magnitude = 0.0
    checks = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip diagonal
            for k in range(n):
                if k == i or k == j:
                    continue  # avoid trivial checks

                checks += 1
                if matrix[i, j] > matrix[i, k] + matrix[k, j]:
                    violation_count += 1
                    violation_magnitude += matrix[i, j] - (matrix[i, k] + matrix[k, j])

    avg_violation = violation_magnitude / violation_count if violation_count else 0
    violation_ratio = violation_count / checks if checks else 0

    return {
        'violation_count': violation_count,
        'total_violation_magnitude': violation_magnitude,
        'average_violation_magnitude': avg_violation,
        'violation_ratio': violation_ratio
    }
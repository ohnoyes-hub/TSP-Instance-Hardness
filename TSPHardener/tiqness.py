import numpy as np
from icecream import ic
from generate_tsp import generate_asymmetric_tsp, generate_euclidean_tsp, triangle_inequality

matrix = generate_asymmetric_tsp(5, 'uniform', 100)
ic(matrix)

ic(generate_euclidean_tsp(5, 'lognormal', 100, 2))

def triangle_inequality_violations(cost_matrix):
    n = cost_matrix.shape[0]
    violations = 0
    severity = []
    total_triplets = n * (n - 1) * (n - 2) // 6

    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if i != k and j != k:
                        if cost_matrix[i, k] > cost_matrix[i, j] + cost_matrix[j, k]:
                            violations += 1
                            severity.append(cost_matrix[i, k] / (cost_matrix[i, j] + cost_matrix[j, k]))

    degree_of_violations = violations / total_triplets
    average_severity = np.mean(severity) if severity else 0

    return degree_of_violations, average_severity

matrix = generate_euclidean_tsp(5, 'uniform', 100, 2)
ic(triangle_inequality(matrix))
ic(triangle_inequality_violations(matrix))

matrix_asym = generate_asymmetric_tsp(5, 'uniform', 100)
ic(triangle_inequality(matrix_asym))
ic(triangle_inequality_violations(matrix_asym))

def test_triangle_inequality(distance_matrix):
    """
    Tests how much the given TSP distance matrix violates the triangle inequality.
    
    Parameters
    ----------
    distance_matrix : list of lists or np.ndarray
        A 2D matrix where distance_matrix[i][j] is the distance from node i to node j.
        
    Returns
    -------
    violations_count : int
        The number of triples (i, j, k) for which the triangle inequality is violated.
    total_violation_magnitude : float
        The sum of how much each violating triple breaks the triangle inequality.
    normalized_violation : float
        The total violation magnitude normalized by the sum of all distances in the matrix.
        If the denominator is zero (e.g., all distances are zero), returns 0.
    """
    n = len(distance_matrix)
    
    # Compute the sum of all distances for normalization
    total_distance_sum = 0.0
    for i in range(n):
        for j in range(n):
            total_distance_sum += distance_matrix[i][j]
    
    # Counters
    violations_count = 0
    total_violation_magnitude = 0.0
    
    # Check all triples (i, j, k) with i, j, k distinct
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                d_ij = distance_matrix[i][j]
                d_ik = distance_matrix[i][k]
                d_kj = distance_matrix[k][j]
                
                # Check the triangle inequality: d(i, j) <= d(i, k) + d(k, j)
                if d_ij > d_ik + d_kj:
                    violation_amount = d_ij - (d_ik + d_kj)
                    violations_count += 1
                    total_violation_magnitude += violation_amount
    
    # Normalize the violation magnitude by the sum of all distances
    if total_distance_sum > 0:
        normalized_violation = total_violation_magnitude / total_distance_sum
    else:
        normalized_violation = 0.0
    
    return violations_count, total_violation_magnitude, normalized_violation

ic(test_triangle_inequality(matrix))
ic(test_triangle_inequality(matrix_asym))
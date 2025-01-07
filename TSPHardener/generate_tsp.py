import numpy as np
from icecream import ic

def generate_asymmetric_tsp(n: int, distribution: str, control: float) -> np.ndarray:
    """
    Generate a random cost matrix of size n x n with values in the range [1, upper) for a asymmetric TSP.
    
    Parameters:
    ----------
    n : int
        The size of the square matrix.
    distribution : str
        The distribution to sample the costs from (either 'uniform' or 'lognormal').
    control : float
        The control parameter for the distribution.
        - For 'uniform', this is the upper bound for the random values.
        - For 'lognormal', this is the sigma parameter.
    """
    if distribution == 'uniform':
        matrix = np.random.random((n, n)) * int(control)
    elif distribution == 'lognormal':
        matrix = (np.random.lognormal(mean=10, sigma=control, size=(n, n))).astype(float) # mean = 10 always in experiments
    else:
        raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
    for i in range(n):
        matrix[i, i] = np.inf
    return matrix

def generate_euclidean_tsp(n: int, distribution: str, control: float, dimensions: int = 100) -> np.ndarray:
    """
    Generate a Euclidean distance matrix for n points in a specified number of dimensions.
    
    Parameters:
    ----------
    n : int
        The number of points.
    distribution : str
        The distribution to sample the points from (either 'uniform' or 'lognormal').
    control : int
        The control parameter for the distribution.
        - For 'uniform', this is the upper bound for the random values.
        - For 'lognormal', this is the sigma parameter.
    dimensions : int
        The dimensionality of the Euclidean space (default is 10).
    Returns:
    -------
    np.ndarray
        A symmetric distance matrix of size n x n.
    """
    # Generate random coordinates for n points in the given dimension
    if distribution == 'uniform':
        points = np.random.random((n, dimensions)) * int(control)
    elif distribution == 'lognormal':
        points = np.random.lognormal(mean=10, sigma=control, size=(n, dimensions))
    else:
        raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    # points = np.random.random((n, dimensions))
    
    # Calculate the pairwise Euclidean distance matrix
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(points[i] - points[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    np.fill_diagonal(distance_matrix, np.inf)  # Set diagonal to infinity for TSP
    
    return distance_matrix

def triangle_inequality(matrix: np.ndarray) -> bool:
        """Test the triangle inequality for a symmetric distance matrix."""
        n = matrix.shape[0]
        # Iterate over all possible triplets (i, j, k) where i, j, k are distinct
        for i in range(n):
            for j in range(n):
                if i == j or matrix[i, j] == np.inf:
                    continue
                for k in range(n):
                    if k == i or k == j or matrix[i, k] == np.inf or matrix[j, k] == np.inf:
                        continue
                    # Check if the triangle inequality holds
                    if matrix[i, j] + matrix[j, k] < matrix[i, k]:
                        ic(f"Triangle inequality failed for indices ({i}, {j}, {k})")
                        return False

        return True


def generate_tsp(city_size, generation_type, distribution, control) -> np.ndarray:
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
    control : float
        The control parameter for the distribution.
        - For the uniform , this is the upper bound for cost values in the matrix.
        - For the lognormal, this is the sigma parameter.
    Returns:
    -------
    np.ndarray
        The generated TSP instance.
    """
    if generation_type == "euclidean":
        return generate_euclidean_tsp(city_size, distribution, control) # dimension of grid is 100x100 unless stated otherwise
    elif generation_type == "asymmetric":
        return generate_asymmetric_tsp(city_size, distribution, control)
    else:
        raise ValueError("Invalid generation type. Choose either 'euclidean' or 'asymmetric'.")
    

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
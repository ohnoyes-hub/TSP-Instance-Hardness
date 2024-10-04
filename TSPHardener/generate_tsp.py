import numpy as np

def generate_asymmetric_tsp(n: int, isUniform: bool, upper: int) -> np.ndarray:
    """
    Generate a random cost matrix of size n x n with values in the range [1, upper) for a asymmetric TSP.
    
    Parameters:
    ----------
    n : int
        The size of the square matrix.
    upper : int
        The upper bound for cost values in the matrix.
    """
    matrix = np.random.random((n, n)) * upper
    for i in range(n):
        matrix[i, i] = np.inf
    return matrix

def generate_euclidean_tsp(n: int, isUniform: bool, dimensions: int = 100) -> np.ndarray:
    """
    Generate a Euclidean distance matrix for n points in a specified number of dimensions.
    
    Parameters:
    ----------
    n : int
        The number of points.
    dimensions : int
        The dimensionality of the Euclidean space (default is 10).
    isUniform : bool
        If True, the matrix is generated with uniform 
        Else, the matrix is generated with random.lognormal
        
    Returns:
    -------
    np.ndarray
        A symmetric distance matrix of size n x n.
    """
    # Generate random coordinates for n points in the given dimension
    if isUniform:
        points = np.random.random((n, dimensions))
    else: # lognormal
        points = np.random.lognormal(mean=0, sigma=1, size=(n, dimensions))
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
                        print(f"Triangle inequality failed for indices ({i}, {j}, {k})")
                        return False

        return True

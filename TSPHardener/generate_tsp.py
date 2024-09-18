import numpy as np

def generate_asymmetric_tsp(n: int, upper: int) -> np.ndarray:
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

def generate_symmetric_matrix(n: int, upper: int) -> np.ndarray:
    """
    Generate a random symmetric distance matrix of size n x n with values in the range [1, upper).
    
    Parameters:
    ----------
    n : int
        The size of the square matrix.
    upper : int
        The upper bound for cost values in the matrix.
    """
    matrix = np.random.random((n, n)) * upper
    matrix = np.triu(matrix)  # Keep upper triangle
    matrix += matrix.T - np.diag(matrix.diagonal())  # Reflect to make symmetric
    np.fill_diagonal(matrix, np.inf)  # Set diagonal to infinity
    return matrix

    # Generate a random symmetric matrix but doesn't follow the triangule inequality
    # matrix = np.random.randint(0, 10, size=(5, 5))
    # matrix = (matrix + matrix.T) // 2

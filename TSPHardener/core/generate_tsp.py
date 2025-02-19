import numpy as np
import logging

logger = logging.getLogger(__name__)

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
                        logger.error(f"Triangle inequality failed for indices ({i}, {j}, {k})")
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
import numpy as np
import logging
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

LOGNORMAL_MEAN = 10

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
        if control <= 0 or not isinstance(control, int):
            raise ValueError("Control parameter must be a positive integer.")
        matrix = np.random.randint(0, control + 1, size=(n, n)).astype(float)
        matrix = _set_diagonal_to_inf(matrix)
    elif distribution == 'lognormal':
        matrix = np.around(np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=control, size=(n, n)))
        matrix = _set_diagonal_to_inf(matrix)
    else:
        raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
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
        The dimensionality of the Cartesian plane (default is 10).
    Returns:
    -------
    np.ndarray
        A symmetric distance matrix of size n x n.
    """
    # Generate random coordinates for n points in the given dimension
    if distribution == 'uniform':
        # scale coordinates to cap at control parameter
        scale = control / np.sqrt(dimensions)
        points = np.random.randint(0, int(scale) + 1, size=(n, dimensions))
    elif distribution == 'lognormal':
        points = np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=control, size=(n, dimensions))
        # scale coordinates to have a mean distance of 10
        scaling_factor = 10 / np.mean(np.linalg.norm(points, axis=1))
        points *= scaling_factor
        points = np.around(points).astype(int)
    else:
        raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
    distances = pdist(points, metric='euclidean')
    distance_matrix = squareform(np.around(distances))
    distance_matrix = _set_diagonal_to_inf(distance_matrix)

    return distance_matrix
    # Calculate the pairwise Euclidean distance matrix
    #distance_matrix = np.zeros((n, n), dtype=float)
    
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         distance = np.linalg.norm(points[i] - points[j])
    #         distance = np.around(distance)  # Around to nearest integer
    #         distance_matrix[i, j] = distance
    #         distance_matrix[j, i] = distance
    
    # np.fill_diagonal(distance_matrix, np.inf)  # Set diagonal to infinity for TSP
    

    #return distance_matrix


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

def _set_diagonal_to_inf(matrix: np.ndarray) -> np.ndarray:
    np.fill_diagonal(matrix, np.inf)
    return matrix

# from icecream import ic
# # # values 
# matrix = generate_tsp(4, "euclidean", "lognormal", 2.4)
# ic(matrix)

# ic(generate_tsp(4, "euclidean", "lognormal", 2.4))
# ic(generate_tsp(4, "asymmetric", "lognormal", 0.2))
# ic(generate_tsp(4, "euclidean", "uniform", 100))
# ic(generate_tsp(4, "asymmetric", "uniform", 100))
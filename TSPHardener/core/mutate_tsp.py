from abc import ABC, abstractmethod
import numpy as np
from .generate_tsp import TSPBuilder

LOGNORMAL_MEAN = 10

def get_mutation_strategy(mutation_type, generation_type, distribution, control):
    """
    Simple factory to create a mutation strategy instance.
    """
    if mutation_type == "swap":
        return SwapMutation()
    elif mutation_type == "inplace":
        return InplaceMutation(distribution, control)
    elif mutation_type == "scramble":
        return ScrambleMutation()
    elif mutation_type == "random_sampling":
        builder = (
            TSPBuilder()
            .set_city_size(control)
            .set_generation_type(generation_type)
            .set_distribution(distribution)
            .set_control(control)
        )
        return RandomSampling(builder)
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    @abstractmethod
    def mutate(self, tsp_instance):
        pass

class SwapMutation(MutationStrategy):
    def mutate(self, tsp_instance):
        if tsp_instance.tsp_type == "euclidean":
            tsp_instance.matrix = swap_mutate_symmetric(tsp_instance.matrix)
        elif tsp_instance.tsp_type == "asymmetric":
            tsp_instance.matrix = swap_mutate(tsp_instance.matrix)
        else:
            raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")

class ScrambleMutation(MutationStrategy):
    def mutate(self, tsp_instance):
        if tsp_instance.tsp_type == 'euclidean':
            tsp_instance.matrix = permute_symmetric_matrix(tsp_instance.matrix)
        else:
            tsp_instance.matrix = permute_matrix(tsp_instance.matrix)
        return tsp_instance

class InplaceMutation(MutationStrategy):
    def __init__(self, distribution, control):
        self.distribution = distribution
        self.control = control
        
    def mutate(self, tsp_instance):
        if tsp_instance.tsp_type == 'euclidean':
            tsp_instance.matrix = mutate_matrix_symmetric(
                self.distribution, tsp_instance.matrix, self.control
            )
        else:
            tsp_instance.matrix = mutate_matrix(
                self.distribution, tsp_instance.matrix, self.control
            )
        return tsp_instance

class RandomSampling(MutationStrategy):
    def __init__(self, tsp_builder):
        self.tsp_builder = tsp_builder
    
    def mutate(self, tsp_instance):
        new_instance = self.tsp_builder.build()
        return new_instance.matrix

#################################################################
# Mutation Core Functions:
#################################################################
def permute_matrix(matrix) -> np.ndarray:
    """
    Generate a random permutation of the given distance matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    """
    n = len(matrix)
    
    matrix = matrix.copy()
    
    # List of indices excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Create a flattened list of values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]
    # print("Indices", values)
    
    # Shuffle the values
    np.random.shuffle(values)
    # print("Shuffled values", values)
    
    # Assign the shuffled values back to the matrix
    for (i, j), value in zip(indices, values):
        matrix[i, j] = value
    
    return matrix

def permute_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Generate a random permutation of the given symmetric cost matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    """
    n = len(matrix)
    
    matrix = matrix.copy()

    # list of indices in the upper triangle excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # upper triangle values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]

    # Shuffle the values to randomize their order
    np.random.shuffle(values)

    # Assign the shuffled values back to the upper triangle
    for (i, j), value in zip(indices, values):
        matrix[i, j] = value
        matrix[j, i] = value  # Mirror the value in the lower triangle
    
    
    return matrix


def swap_mutate(matrix) -> np.ndarray:
    """
    Swap two random elements in the cost matrix excluding the diagonal.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to mutate.
    """
    arr = matrix.copy()
    rows, cols = arr.shape
    total_elements = rows * cols
    
    def get_off_diag_index():
        while True:
            idx = np.random.randint(total_elements)
            i = idx // cols
            j = idx % cols
            if i != j:
                return (i, j)
    
    i1, j1 = get_off_diag_index()

    # second off-diagonal index
    while True:
        i2, j2 = get_off_diag_index()
        if (i1, j1) != (i2, j2):
            break

    # Swap values
    arr[i1][j1], arr[i2][j2] = arr[i2][j2], arr[i1][j1]
    return arr

def swap_mutate_symmetric(matrix) -> np.ndarray:
    """
    Swap two random elements in a Eucliean TSP excluding the diagonal.

    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to mutate.
    """
    arr = matrix.copy()
    n = arr.shape[0]

    if n < 3:
        raise ValueError("City size must be at least 3 or greater for swapping.")
    
    def get_upper_triangle_index():
        i = np.random.randint(n - 1)
        j = np.random.randint(i + 1, n)
        return (i, j)
    
    i1, j1 = get_upper_triangle_index()
    
    while True:
        i2, j2 = get_upper_triangle_index()
        if (i2, j2) != (i1, j1):
            break
    
    # Swap upper triangle and mirror to lower
    arr[i1, j1], arr[i2, j2] = arr[i2, j2], arr[i1, j1]
    arr[j1, i1] = arr[i1, j1]
    arr[j2, i2] = arr[i2, j2]
    
    return arr
 
def mutate_matrix(distribution, _matrix, _control):
    """
    Mutate a random element in the cost matrix excluding the diagonal.
    """
    matrix = _matrix.copy()
    n = matrix.shape[0]
    number1, number2 = 0, 0

    # ensure that the random city pair to mutate is not on the diagonal
    while number1 == number2:
        number1, number2 = np.random.randint(0, n), np.random.randint(0, n)
    previous_number = matrix[number1,number2]
    
    while matrix[number1,number2] == previous_number:
        if distribution == "uniform":
            matrix[number1,number2] = np.random.randint(1,_control)
        elif distribution == "lognormal":
            matrix[number1,number2] = np.around(np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=_control))
        else:
            raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
    return matrix

def mutate_matrix_symmetric(distribution, _matrix, _upper):
    matrix = _matrix.copy()
    n = matrix.shape[0]
    number1, number2 = 0, 0

    while number1 == number2:
        number1, number2 = np.random.randint(0,n), np.random.randint(0,n)
    previous_number = matrix[number1,number2]

    while matrix[number1,number2] == previous_number:
        if distribution == "uniform":
            new_val = np.random.randint(1, _upper)
            matrix[number1, number2] = new_val
            matrix[number2, number1] = new_val  # Symmetric update
        elif distribution == "lognormal":
            new_val = np.around(np.random.lognormal(mean=10, sigma=_upper))
            matrix[number1, number2] = new_val
            matrix[number2, number1] = new_val
        else:
            raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
    return matrix


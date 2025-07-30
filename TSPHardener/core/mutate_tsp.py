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
    elif mutation_type == "wouter": # wouter = inplace
        return InplaceMutation(distribution, control)
    elif mutation_type == "scramble":
        return ScrambleMutation()
    elif mutation_type == "random_sampling":
        builder = (
            TSPBuilder()
            .set_generation_type(generation_type)
            .set_distribution(distribution)
            .set_control(control)
        )
        return RandomSampling(builder)
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

##################################################################
# Mutation Strategy:
##################################################################
class MutationStrategy(ABC):
    """
    Abstract class for mutation strategies.
    """
    @abstractmethod
    def mutate(self, tsp_instance) -> None:
        """Mutate the given TSP instance's distance matrix."""

class SwapMutation(MutationStrategy):
    """
    Swaps two random non-diagonal elements in the distance matrix.
    """
    def mutate(self, tsp_instance):
        if tsp_instance.tsp_type == "euclidean":
            tsp_instance.matrix = swap_mutate_symmetric(tsp_instance.matrix)
        elif tsp_instance.tsp_type == "asymmetric":
            tsp_instance.matrix = swap_mutate(tsp_instance.matrix)
        else:
            raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")
        return tsp_instance

class ScrambleMutation(MutationStrategy):
    """
    Randomly permute all non-diagonal elements in the distance matrix.
    """
    def mutate(self, tsp_instance):
        if tsp_instance.tsp_type == 'euclidean':
            tsp_instance.matrix = permute_symmetric_matrix(tsp_instance.matrix)
        else:
            tsp_instance.matrix = permute_matrix(tsp_instance.matrix)
        return tsp_instance

class InplaceMutation(MutationStrategy):
    """Mutates a single element (or symmetric pair) using the specified distribution."""
    def __init__(self, distribution, control):
        if distribution not in ["uniform", "lognormal"]:
            raise ValueError(f"Invalid distribution: {distribution}. Choose 'uniform' or 'lognormal'.")
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
    """Replaces the matrix with a new instance built by the TSPBuilder."""
    def __init__(self, tsp_builder):
        self.tsp_builder = tsp_builder
    
    def mutate(self, tsp_instance):
        tsp_instance.matrix = self.tsp_builder.build().matrix
        return tsp_instance

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
    # get list of all indices excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(n) if i != j]   
    # Shuffle the indices
    np.random.shuffle(indices)
    # Create a flattened list of values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]    
    # Shuffle the values
    np.random.shuffle(values)
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
    # get random index excluding the diagonal
    def get_off_diag_index():
        while True:
            idx = np.random.randint(total_elements)
            i = idx // cols
            j = idx % cols
            if i != j:
                return (i, j)
    
    i1, j1 = get_off_diag_index()

    # get second off-diagonal index
    while True:
        i2, j2 = get_off_diag_index()
        if (i1, j1) != (i2, j2):
            break

    # Swap indices
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

# import numpy as np

# LOGNORMAL_MEAN = 10

# def permute_matrix(matrix) -> np.ndarray:
#     """
#     Generate a random permutation of the given distance matrix.
    
#     Parameters:
#     ----------
#     matrix : np.ndarray
#         The distance matrix to permute.
#     """
#     n = len(matrix)
    
#     matrix = matrix.copy()
    
#     # List of indices excluding the diagonal
#     indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    
#     # Shuffle the indices
#     np.random.shuffle(indices)
    
#     # Create a flattened list of values excluding the diagonal
#     values = [matrix[i, j] for i, j in indices]
#     # print("Indices", values)
    
#     # Shuffle the values
#     np.random.shuffle(values)
#     # print("Shuffled values", values)
    
#     # Assign the shuffled values back to the matrix
#     for (i, j), value in zip(indices, values):
#         matrix[i, j] = value
    
#     return matrix

# def permute_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
#     """
#     Generate a random permutation of the given symmetric cost matrix.
    
#     Parameters:
#     ----------
#     matrix : np.ndarray
#         The distance matrix to permute.
#     """
#     n = len(matrix)
    
#     matrix = matrix.copy()

#     # list of indices in the upper triangle excluding the diagonal
#     indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

#     # upper triangle values excluding the diagonal
#     values = [matrix[i, j] for i, j in indices]

#     # Shuffle the values to randomize their order
#     np.random.shuffle(values)

#     # Assign the shuffled values back to the upper triangle
#     for (i, j), value in zip(indices, values):
#         matrix[i, j] = value
#         matrix[j, i] = value  # Mirror the value in the lower triangle
    
    
#     return matrix


# def swap_mutate(matrix) -> np.ndarray:
#     """
#     Swap two random elements in the cost matrix excluding the diagonal.
    
#     Parameters:
#     ----------
#     matrix : np.ndarray
#         The distance matrix to mutate.
#     """
#     arr = matrix.copy()
#     rows, cols = arr.shape
#     total_elements = rows * cols
    
#     def get_off_diag_index():
#         while True:
#             idx = np.random.randint(total_elements)
#             i = idx // cols
#             j = idx % cols
#             if i != j:
#                 return (i, j)
    
#     i1, j1 = get_off_diag_index()

#     # second off-diagonal index
#     while True:
#         i2, j2 = get_off_diag_index()
#         if (i1, j1) != (i2, j2):
#             break

#     # Swap values
#     arr[i1][j1], arr[i2][j2] = arr[i2][j2], arr[i1][j1]
#     return arr

# def swap_mutate_symmetric(matrix) -> np.ndarray:
#     """
#     Swap two random elements in a Eucliean TSP excluding the diagonal.

#     Parameters:
#     ----------
#     matrix : np.ndarray
#         The distance matrix to mutate.
#     """
#     arr = matrix.copy()
#     n = arr.shape[0]

#     if n < 3:
#         raise ValueError("City size must be at least 3 or greater for swapping.")
    
#     def get_upper_triangle_index():
#         i = np.random.randint(n - 1)
#         j = np.random.randint(i + 1, n)
#         return (i, j)
    
#     i1, j1 = get_upper_triangle_index()
    
#     while True:
#         i2, j2 = get_upper_triangle_index()
#         if (i2, j2) != (i1, j1):
#             break
    
#     # Swap upper triangle and mirror to lower
#     arr[i1, j1], arr[i2, j2] = arr[i2, j2], arr[i1, j1]
#     arr[j1, i1] = arr[i1, j1]
#     arr[j2, i2] = arr[i2, j2]
    
#     return arr
 

# def mutate_matrix(distribution, _matrix, _control):
#     matrix = _matrix.copy()
#     n = matrix.shape[0]
#     number1, number2 = 0, 0

#     # ensure that the random city pair to mutate is not on the diagonal
#     while number1 == number2:
#         number1, number2 = np.random.randint(0, n), np.random.randint(0, n)
#     previous_number = matrix[number1,number2]
    
#     while matrix[number1,number2] == previous_number:
#         if distribution == "uniform":
#             matrix[number1,number2] = np.random.randint(1,_control)
#         elif distribution == "lognormal":
#             matrix[number1,number2] = np.around(np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=_control))
#         else:
#             raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
#     return matrix

# def mutate_matrix_symmetric(distribution, _matrix, _upper):
#     matrix = _matrix.copy()
#     n = matrix.shape[0]
#     number1, number2 = 0, 0

#     while number1 == number2:
#         number1, number2 = np.random.randint(0,n), np.random.randint(0,n)
#     previous_number = matrix[number1,number2]

#     while matrix[number1,number2] == previous_number:
#         if distribution == "uniform":
#             new_val = np.random.randint(1, _upper)
#             matrix[number1, number2] = new_val
#             matrix[number2, number1] = new_val  # Symmetric update
#         elif distribution == "lognormal":
#             new_val = np.around(np.random.lognormal(mean=10, sigma=_upper))
#             matrix[number1, number2] = new_val
#             matrix[number2, number1] = new_val
#         else:
#             raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
#     return matrix

# def shuffle(tsp_type, matrix):
#     """
#     Shuffle the elements of either a symmetric or asymmetric TSP matrix.

#     Parameters:
#     ----------
#     tsp_type : str
#         The type of TSP matrix ('euclidean' or 'asymmetric').
#     matrix : np.ndarray
#         The distance matrix to shuffle.
#     """
#     if tsp_type == 'euclidean':
#         return permute_symmetric_matrix(matrix)
#     elif tsp_type == 'asymmetric':
#         return permute_matrix(matrix)
#     else:
#         raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")
    
# def mutate(distribution, tsp_type, matrix, upper):
#     """
#     Mutate the elements of either a euclidean or asymmetric TSP matrix.

#     Parameters:
#     ----------
#     distribution : str
#         The distribution to sample the points from (either 'uniform' or 'lognormal').
#     tsp_type : str
#         The type of TSP matrix ('euclidean' or 'asymmetric').
#     matrix : np.ndarray
#         The distance matrix to mutate.
#     upper : int
#         The upper bound for the random mutation.
#     """
#     if tsp_type == 'euclidean':
#         return mutate_matrix_symmetric(distribution, matrix, upper)
#     elif tsp_type == 'asymmetric':
#         return mutate_matrix(distribution, matrix, upper)
#     else:
#         raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")

# def swap(tsp_type, matrix):
#     """
#     Swap two random elements in either a euclidean or asymmetric TSP matrix.

#     Parameters:
#     ----------
#     tsp_type : str
#         The type of TSP matrix ('euclidean' or 'asymmetric').
#     matrix : np.ndarray
#         The distance matrix to mutate.
#     """
#     if tsp_type == "euclidean":
#         return swap_mutate_symmetric(matrix)
#     elif tsp_type == "asymmetric":
#         return swap_mutate(matrix)
#     else:
#         raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")
    

# def apply_mutation(matrix, mutation_type, tsp_type, control, distribution):
#     """
#     Apply a mutation to the given TSP instance.
    
#     Parameters:
#     ----------
#     matrix : np.ndarray
#         The TSP instance to mutate.
#     mutation_type : str
#         The mutation strategy to apply (swap, scramble, or Wouter's mutation).
#     tsp_type : str
#         The type of TSP instance (euclidean or asymmetric).
#     control : float
#         The upper bound for cost values in the matrix.
        
#     Returns:
#     -------
#     np.ndarray
#         The mutated TSP instance.
#     """
#     if mutation_type == "swap":
#         return swap(tsp_type, matrix)
#     elif mutation_type == "scramble":
#         return shuffle(tsp_type, matrix)
#     elif mutation_type == "wouter":
#         return mutate(distribution, tsp_type, matrix, control)
#     else:
#         raise ValueError("Invalid mutation type. Choose either 'swap', 'scramble', or 'wouter'.")
    

# # from .generate_tsp import generate_asymmetric_tsp, generate_euclidean_tsp
# # asy_matrix = generate_asymmetric_tsp(5, 'uniform', 20)
# # eu_matrix = generate_euclidean_tsp(5, 'uniform', 100)

# # from icecream import ic
# # ic(asy_matrix, permute_matrix(asy_matrix))
# # ic(eu_matrix, permute_symmetric_matrix(eu_matrix))

# # ic(eu_matrix, swap_mutate_symmetric(eu_matrix))
# # ic(eu_matrix, mutate_matrix_symmetric('uniform', eu_matrix, 100))
# # ic(asy_matrix, swap_mutate(asy_matrix))
# # ic(asy_matrix, mutate_matrix('uniform', asy_matrix, 100))
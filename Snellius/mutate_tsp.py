import numpy as np
from icecream import ic

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
    matrix = matrix.copy()
    
    n = len(matrix)
    # swap index (i,j) with (k, l)
    i, j, k, l = np.random.randint(0, n), np.random.randint(0, n), np.random.randint(0, n), np.random.randint(0, n)
    
    # Ensure that indices are not the same
    while i == k and j == l:
        k, l = np.random.randint(0, n), np.random.randint(0, n)
        # Ensure that the indices are not diagonal
        while i == j or k == l:
            j = np.random.randint(0, n)
            k = np.random.randint(0, n)
    
    # Swap the elements
    # temp = matrix[i, j]
    matrix[i, j], matrix[k, l] = matrix[k, l], matrix[i, j]

    return matrix

def swap_mutate_symmetric(matrix) -> np.ndarray:
    """
    Swap two random elements in a Eucliean TSP excluding the diagonal.

    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to mutate.
    """
    matrix = matrix.copy()
    
    n = len(matrix)
    
    while True:
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        k, l = np.random.randint(0, n), np.random.randint(0, n)
        # Ensure that indices are not the same
        while i == k and j == l:
            k, l = np.random.randint(0, n), np.random.randint(0, n)
            # Ensure that the indices are not diagonal
            while i == j or k == l:
                j = np.random.randint(0, n)
                k = np.random.randint(0, n)
        
        # Ensure that the elements are in the upper triangle (i < j and k < l)
        if i < j and k < l and (i != k or j != l):
            break
        
    # Swap the elements
    matrix[i, j], matrix[k, l] = matrix[k, l], matrix[i, j]
    matrix[j, i], matrix[l, k] = matrix[l, k], matrix[j, i]  # Mirror the value in the lower triangle
    # print(f"Swapped ({i}, {j}) with ({k}, {l})")
    return matrix
 

def mutate_matrix(distribution, _matrix, _control):
    matrix = _matrix.copy()
    number1, number2 = 0, 0

    while number1 == number2:
        number1, number2 = np.random.randint(0,matrix.shape[0]), np.random.randint(0,matrix.shape[0])
    previous_number = matrix[number1,number2]
    while matrix[number1,number2] == previous_number:
        if distribution == "uniform":
            matrix[number1,number2] = np.random.randint(1,_control)
        elif distribution == "lognormal":
            matrix[number1,number2] = np.random.lognormal(mean=10, sigma=_control)
        else:
            raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    return matrix

def mutate_matrix_symmetric(distribution, _matrix, _upper):
    matrix = _matrix.copy()
    number1, number2 = 0, 0

    while number1 == number2:
        number1, number2 = np.random.randint(0,matrix.shape[0]), np.random.randint(0,matrix.shape[0])
    previous_number = matrix[number1,number2]
    while matrix[number1,number2] == previous_number:
        if distribution == "uniform":
            matrix[number1,number2] = np.random.randint(1,_upper)
            matrix[number2,number1] = matrix[number1,number2] # Mirror the value in the lower triangle
        elif distribution == "lognormal":
            matrix[number1,number2] = np.random.lognormal(mean=10, sigma=_upper)
            matrix[number2,number1] = matrix[number1,number2]
        else:
            raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    return matrix

def shuffle(tsp_type, matrix):
    """
    Shuffle the elements of either a symmetric or asymmetric TSP matrix.

    Parameters:
    ----------
    tsp_type : str
        The type of TSP matrix ('euclidean' or 'asymmetric').
    matrix : np.ndarray
        The distance matrix to shuffle.
    """
    if tsp_type == 'euclidean':
        return permute_symmetric_matrix(matrix)
    elif tsp_type == 'asymmetric':
        return permute_matrix(matrix)
    else:
        raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")
    
def mutate(distribution, tsp_type, matrix, upper):
    """
    Mutate the elements of either a euclidean or asymmetric TSP matrix.

    Parameters:
    ----------
    distribution : str
        The distribution to sample the points from (either 'uniform' or 'lognormal').
    tsp_type : str
        The type of TSP matrix ('euclidean' or 'asymmetric').
    matrix : np.ndarray
        The distance matrix to mutate.
    upper : int
        The upper bound for the random mutation.
    """
    if tsp_type == 'euclidean':
        return mutate_matrix_symmetric(distribution, matrix, upper)
    elif tsp_type == 'asymmetric':
        return mutate_matrix(distribution, matrix, upper)
    else:
        raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")

def swap(tsp_type, matrix):
    """
    Swap two random elements in either a euclidean or asymmetric TSP matrix.

    Parameters:
    ----------
    tsp_type : str
        The type of TSP matrix ('euclidean' or 'asymmetric').
    matrix : np.ndarray
        The distance matrix to mutate.
    """
    if tsp_type == "euclidean":
        return swap_mutate_symmetric(matrix)
    elif tsp_type == "asymmetric":
        return swap_mutate(matrix, )
    else:
        raise ValueError("Invalid TSP type. Choose either 'euclidean' or 'asymmetric'.")
    

def apply_mutation(matrix, mutation_type, tsp_type, control, distribution):
    """
    Apply a mutation to the given TSP instance.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The TSP instance to mutate.
    mutation_type : str
        The mutation strategy to apply (swap, scramble, or Wouter's mutation).
    tsp_type : str
        The type of TSP instance (euclidean or asymmetric).
    control : float
        The upper bound for cost values in the matrix.
        
    Returns:
    -------
    np.ndarray
        The mutated TSP instance.
    """
    if mutation_type == "swap":
        return swap(tsp_type, matrix)
    elif mutation_type == "scramble":
        return shuffle(tsp_type, matrix)
    elif mutation_type == "wouter":
        return mutate(distribution, tsp_type, matrix, control)
    else:
        raise ValueError("Invalid mutation type. Choose either 'swap', 'scramble', or 'wouter'.")
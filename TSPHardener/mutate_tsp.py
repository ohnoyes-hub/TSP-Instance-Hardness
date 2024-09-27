import numpy as np

def permute_matrix(matrix, _print) -> np.ndarray:
    """
    Generate a random permutation of the given distance matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    _print : bool
        Print results in the console if True.
    """
    n = len(matrix)
    
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
    permuted_matrix = matrix.copy()
    
    # TODO: find a faster way to permute the matrix. This is essentially brute force O(n^2 - n) where we shuffle list of n^2 - n elements in n x n matrix.
    # List of indices excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(n) if i != j]

    if _print:
        print(f"Indices to shuffle: {indices}")
    
    # Shuffle the indices
    np.random.shuffle(indices)
        
    if _print:
        print(f"Shuffled indices: {indices}")
    
    # Create a flattened list of values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]
    
    # Shuffle the values
    np.random.shuffle(values)
    
    # Assign the shuffled values back to the matrix
    for (i, j), value in zip(indices, values):
        permuted_matrix[i, j] = value
    
    if _print:
        print(f"Permuted matrix:\n{permuted_matrix.round(1)}")
    
    return permuted_matrix

def permute_symmetric_matrix(matrix: np.ndarray, _print) -> np.ndarray:
    """
    Generate a random permutation of the given symmetric cost matrix.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to permute.
    _print : bool
        Print results in the console if True.
    """
    n = len(matrix)
    
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
    permuted_matrix = matrix.copy()

    # list of indices in the upper triangle excluding the diagonal
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # upper triangle values excluding the diagonal
    values = [matrix[i, j] for i, j in indices]

    # Shuffle the values to randomize their order
    np.random.shuffle(values)

    # Assign the shuffled values back to the upper triangle
    for (i, j), value in zip(indices, values):
        permuted_matrix[i, j] = value
        permuted_matrix[j, i] = value  # Mirror the value in the lower triangle
    
    if _print:
        print(f"Permuted matrix:\n{permuted_matrix.round(1)}")
    
    return permuted_matrix


def swap_mutate(matrix, _print) -> np.ndarray:
    """
    Swap two random elements in the cost matrix excluding the diagonal.
    
    Parameters:
    ----------
    matrix : np.ndarray
        The distance matrix to mutate.
    _print : bool
        Print results in the console if True.
    """
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}")
    
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
    
    if _print:
        print(f"Swapping elements at indices ({i}, {j}) and ({k}, {l})")
    
    # Swap the elements
    temp = matrix[i, j]
    matrix[i, j], matrix[k, l] = matrix[k, l], matrix[i, j]

    if _print:
        print(f"Mutated matrix:\n{matrix.round(1)}")
    
    return matrix

def mutate_matrix(_matrix, _upper, _print):
    matrix = _matrix.copy()
    number1, number2 = 0, 0

    while number1 == number2:
        number1, number2 = np.random.randint(0,matrix.shape[0]), np.random.randint(0,matrix.shape[0])
    previous_number = matrix[number1,number2]
    while matrix[number1,number2] == previous_number:
        matrix[number1,number2] = np.random.randint(1,_upper)
    if _print:
        print(_matrix[number1,number2].round(1), "at", (number1,number2), "becomes", matrix[number1,number2].round(1))

    return matrix
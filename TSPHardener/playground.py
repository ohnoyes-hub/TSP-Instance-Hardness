from typing import List

# This file is where I write and test code snippets that I want to use in the main codebase.


def permutations(A: List[int]) -> List[List[int]]:
    def directed_permutations(i):
        if i == len(A) - 1:
            result.append(A.copy())
            return

        # Try every possibility for A[i].
        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            # Generate all permutations for A[i + 1:].
            directed_permutations(i + 1)
            A[i], A[j] = A[j], A[i]

    result: List[List[int]] = []
    directed_permutations(0)
    return result

# Variant 3: Generate permuations row and column wise of a 2D array.
def permute_2d_array(A: List[List[int]]) -> List[List[List[int]]]:
    def directed_permutations(i):
        if i == len(A) - 1:
            result.append(A.copy())
            return

        # Try every possibility for A[i].
        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            # Generate all permutations for A[i + 1:].
            directed_permutations(i + 1)
            A[i], A[j] = A[j], A[i]

    result: List[List[List[int]]] = []
    directed_permutations(0)
    return result


def compute_random_permutation(n: int) -> List[int]:
    permutation = list(range(n))
    random_sampling(n, permutation)
    return permutation

# Variant 1: Randomly permute an array.
def random_sampling_array(A: List[int]) -> List[int]:
    for i in range(len(A)):
        r = random.randint(i, len(A) - 1)
        A[i], A[r] = A[r], A[i]
    return A

# Variant 2: Randomly permute a 2D array.
def random_sampling_2d_array(A: List[List[int]]) -> List[List[int]]:
    for i in range(len(A)):
        for j in range(len(A[i])):
            r = random.randint(i, len(A) - 1)
            c = random.randint(j, len(A[i]) - 1)
            A[i][j], A[r][j] = A[r][j], A[i][j]
            A[i][j], A[i][c] = A[i][c], A[i][j]
    return A

import numpy as np
# Variant 4: Randomly permute a 2D array using numpy.
def random_sampling_2d_array_numpy(A: List[List[int]]) -> List[List[int]]:
    for i in range(len(A)):
        for j in range(len(A[i])):
            r = np.random.randint(i, len(A) - 1)
            c = np.random.randint(j, len(A[i]) - 1)
            A[i][j], A[r][j] = A[r][j], A[i][j]
            A[i][j], A[i][c] = A[i][c], A[i][j]

import random
def random_permute_numpy(A: List[int]) -> None:
    for i in range(len(A)):
        r = np.random.randint(i, len(A) - 1)
        A[i], A[r] = A[r], A[i]

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

# Optimize: mutate/permuate the whole matrix.
def mutate_matrix_optimize(matrix, upper, _print) -> np.ndarray:
    n = len(matrix)
    if _print:
        print(f"Original matrix:\n{matrix.round(1)}" + " becomes: ", end="")

    for i in range(n -1):
        for j in range(n - 1):
            # skip the diagonal elements
            if i == j:
                continue
            r = np.random.randint(i, n - 1)
            c = np.random.randint(j, n - 1)
            # Ensure that random row and column are not on the diagonal
            if r == c:
                continue
            else:    
                matrix[i][j], matrix[r][j] = matrix[r][j], matrix[i][j]
                matrix[i][j], matrix[i][c] = matrix[i][c], matrix[i][j]

    if _print:
        print(f"\n{matrix.round(1)}")
    return matrix

if __name__ == "__main__":
    array = [1, 2, 3]
    array3 = [[1, 2], [3, 4], [5, 6]]

    print(print(array))

    print(random_sampling_array(array))
    print(random_sampling_2d_array(array3))

    arr = [2,3,4,5,6,7,8,9,10]
    print(random_sampling_array(arr))

    arr_numpy = np.array([2,3,4,5,6,7,8,9,10])
    arr_numpy_2d = np.array([[np.infty,3,4,4], [5,np.infty,3,4], [8,9,np.infty,6], [8,9,6,np.infty]])

    #mutate_matrix(arr_numpy, 10, True)
    #mutate_matrix(arr_numpy_2d, 3, True)
    print(arr_numpy_2d.shape)
    print(arr_numpy_2d.shape[0])
    print(arr_numpy_2d.shape[1])
    print(len(arr_numpy_2d))
    print(len(arr_numpy_2d[0]))
    mutate_matrix(arr_numpy_2d, 10, True)
    mutate_matrix_optimize(arr_numpy_2d, 10, True)

    matrix = np.array([[0, 1, 2],
                   [1, 0, 3],
                   [2, 3, 0]])

    print("Original matrix:")
    print(matrix)

    result1 = mutate_off_diagonal_symmetric(matrix.copy(), 999)
    print("\nMatrix after mutation with value 999:")
    print(result1)

    result2 = mutate_off_diagonal_symmetric(matrix.copy(), -1)
print("\nMatrix after mutation with value -1:")
print(result2)

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0

        for i in range(len(nums)):
            if nums[i] != val:
                # partition
                nums[k] = nums[i]
                k += 1
        return k
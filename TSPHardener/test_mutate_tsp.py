import unittest
import numpy as np
from mutate_tsp import permute_matrix, permute_symmetric_matrix, swap_mutate, mutate_matrix, mutate_matrix_symmetric, swap_mutate_symmetric

class TestMutateTSP(unittest.TestCase):

    def setUp(self):
        # Set up a sample distance matrix for testing
        self.matrix = np.array([[0, 2, 3],
                                [2, 0, 4],
                                [3, 4, 0]])

    def test_permute_matrix(self):
        permuted = permute_matrix(self.matrix)
        # Check if the shape remains the same
        self.assertEqual(permuted.shape, self.matrix.shape)
        # Check if the diagonal is unchanged
        np.testing.assert_array_equal(np.diag(permuted), np.diag(self.matrix))

    def test_permute_symmetric_matrix(self):
        permuted = permute_symmetric_matrix(self.matrix)
        # Check if the shape remains the same
        self.assertEqual(permuted.shape, self.matrix.shape)
        # Check if the matrix is still symmetric
        self.assertTrue((permuted == permuted.T).all())
        # Check if the diagonal is unchanged
        np.testing.assert_array_equal(np.diag(permuted), np.diag(self.matrix))

    def test_swap_mutate(self):
        mutated = swap_mutate(self.matrix.copy(), _print=False)
        # Check if the shape remains the same
        self.assertEqual(mutated.shape, self.matrix.shape)
        # Ensure only off-diagonal elements have changed
        for i in range(self.matrix.shape[0]):
            self.assertEqual(mutated[i, i], self.matrix[i, i])

    def test_mutate_matrix(self):
        upper = 10
        mutated = mutate_matrix(self.matrix.copy(), upper)
        # Check if the shape remains the same
        self.assertEqual(mutated.shape, self.matrix.shape)
        # Ensure at least one element has changed
        self.assertFalse(np.array_equal(mutated, self.matrix))

    def test_mutate_matrix_symmetric(self):
        upper = 10
        mutated = mutate_matrix_symmetric(self.matrix.copy(), upper)
        # Check if the shape remains the same
        self.assertEqual(mutated.shape, self.matrix.shape)
        # Check if the matrix is still symmetric
        self.assertTrue((mutated == mutated.T).all())
        # Ensure at least one element has changed
        self.assertFalse(np.array_equal(mutated, self.matrix))

    def test_swap_mutate_symmetric(self):
        mutated = swap_mutate_symmetric(self.matrix.copy())
        # Check if the shape remains the same
        self.assertEqual(mutated.shape, self.matrix.shape)
        # Ensure only off-diagonal elements have changed
        for i in range(self.matrix.shape[0]):
            self.assertEqual(mutated[i, i], self.matrix[i, i])
        # Check if the matrix is still symmetric
        self.assertTrue((mutated == mutated.T).all())


if __name__ == '__main__':
    unittest.main()

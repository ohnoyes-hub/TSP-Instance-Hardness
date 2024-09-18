import unittest
import numpy as np
from generate_tsp import generate_symmetric_matrix

class TestGenerateSymmetricMatrix(unittest.TestCase):
    
    def test_matrix_size(self):
        n = 5
        upper = 10
        matrix = generate_symmetric_matrix(n, upper)
        self.assertEqual(matrix.shape, (n, n), "Matrix size is incorrect")
    
    def test_matrix_symmetry(self):
        n = 5
        upper = 10
        matrix = generate_symmetric_matrix(n, upper)
        self.assertTrue(np.allclose(matrix, matrix.T, atol=1e-8), "Matrix is not symmetric")
    
    def test_matrix_diagonal(self):
        n = 5
        upper = 10
        matrix = generate_symmetric_matrix(n, upper)
        self.assertTrue(np.all(np.isinf(np.diag(matrix))), "Diagonal elements are not infinity")
    
    def test_matrix_value_range(self):
        n = 5
        upper = 10
        matrix = generate_symmetric_matrix(n, upper)
        mask = ~np.isinf(matrix)
        self.assertTrue(np.all(matrix[mask] >= 0) and np.all(matrix[mask] < upper), "Matrix values are out of range")
    
    

if __name__ == '__main__':
    unittest.main()
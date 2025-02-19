import unittest
import numpy as np
from TSPHardener.core.generate_tsp import generate_asymmetric_tsp, generate_euclidean_tsp, triangle_inequality

class TestGenerateTSP(unittest.TestCase):

    def test_generate_asymmetric_tsp(self):
        """Test the generate_asymmetric_tsp function for basic properties."""
        n = 5
        upper = 100
        isUniform = True
        
        # Generate the asymmetric TSP matrix
        matrix = generate_asymmetric_tsp(n, isUniform, upper)
        
        # Test the shape of the generated matrix
        self.assertEqual(matrix.shape, (n, n))
        
        # Test that diagonal elements are set to infinity
        for i in range(n):
            self.assertEqual(matrix[i, i], np.inf)
        
        # Test that the matrix is not symmetric
        symmetric = np.allclose(matrix, matrix.T)
        self.assertFalse(symmetric, "The generated matrix should be asymmetric.")

        # Test that elements are in the expected range
        self.assertTrue(np.all((matrix < upper) | (matrix == np.inf)))
        self.assertTrue(np.all(matrix >= 0))

    def test_generate_euclidean_tsp(self):
        """Test the generate_euclidean_tsp function for basic properties."""
        n = 5
        dimensions = 2
        isUniform = True
        
        # Generate the Euclidean TSP matrix
        matrix = generate_euclidean_tsp(n, isUniform, dimensions)
        
        # Test the shape of the generated matrix
        self.assertEqual(matrix.shape, (n, n))
        
        # Test that diagonal elements are set to infinity
        for i in range(n):
            self.assertEqual(matrix[i, i], np.inf)
        
        # Test that the matrix is symmetric
        symmetric = np.allclose(matrix, matrix.T)
        self.assertTrue(symmetric, "The generated matrix should be symmetric.")
        
        # Test that distances are non-negative
        self.assertTrue(np.all(matrix[matrix != np.inf] >= 0))

    def test_triangle_inequality(self):
        """Test the triangle inequality for a generated symmetric matrix."""
        n = 5
        dimensions = 2
        isUniform = True
        
        # Generate a symmetric Euclidean TSP matrix
        matrix = generate_euclidean_tsp(n, isUniform, dimensions)
        
        # Test the triangle inequality
        self.assertTrue(triangle_inequality(matrix))

    
if __name__ == '__main__':
    unittest.main()

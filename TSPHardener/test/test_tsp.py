import unittest
import numpy as np
from core.generate_tsp import TSPBuilder
from core.mutate_tsp import SwapMutation, ScrambleMutation, InplaceMutation, RandomSampling
from scipy.stats import lognorm, chisquare

class TestTSPGeneration(unittest.TestCase):

    def test_asymmetric_tsp_uniform(self):
        tsp = TSPBuilder().set_city_size(5).set_generation_type("asymmetric") \
                          .set_distribution("uniform").set_control(100).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertFalse(np.allclose(tsp.matrix, tsp.matrix.T))
        # Ensure values are within uniform distribution bounds (0-100)
        mask = ~np.eye(tsp.matrix.shape[0], dtype=bool)  # Exclude diagonal
        assert np.all((tsp.matrix[mask] >= 0) & (tsp.matrix[mask] <= 100)), "Values out of bounds."

    def test_euclidean_tsp_lognormal(self):
        tsp = TSPBuilder().set_city_size(5).set_generation_type("euclidean") \
                          .set_distribution("lognormal").set_control(0.5).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertTrue(np.allclose(tsp.matrix, tsp.matrix.T))
        

class TestATSPMutation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.tsp_instance = TSPBuilder().set_city_size(4).set_generation_type("asymmetric").set_distribution("uniform").set_control(20).build()
    
    def test_swap_mutation(self):
        original_matrix = self.tsp_instance.matrix.copy()
        SwapMutation().mutate(self.tsp_instance)
        self.assertEqual(self.tsp_instance.matrix.shape, (4, 4))
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        self.assertEqual(differences, 2) # only two elements should be swapped
        # -- Test for multiple runs of muatation will not swap the same elements repeatedly

    def test_scramble_mutation(self):
        base = self.tsp_instance.matrix.copy()
        ScrambleMutation().mutate(self.tsp_instance)
        self.assertEqual(self.tsp_instance.matrix.shape, (4, 4))
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, base))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))

    def test_inplace_mutation_uniform(self):
        original_matrix = self.tsp_instance.matrix.copy()
        InplaceMutation('uniform', 20).mutate(self.tsp_instance)
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        self.assertEqual(differences, 1)  # only one element changed

class TestETSPMutation(unittest.TestCase):

    def setUp(self):
        np.random.seed(58)
        self.tsp_instance = TSPBuilder().set_city_size(6).set_generation_type("euclidean").set_distribution("uniform").set_control(20).build()
    
    def test_swap_mutation(self):
        original_matrix = self.tsp_instance.matrix.copy()
        SwapMutation().mutate(self.tsp_instance)
        self.assertEqual(self.tsp_instance.matrix.shape, (6, 6))
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        # Swap mutation should change two elements in the matrix
        # In symmetric TSP distance matrices this means the upper triangle swap should match with the lower triangle
        self.assertEqual(differences, 4)

    def test_scramble_mutation(self):
        base = self.tsp_instance.matrix.copy()
        ScrambleMutation().mutate(self.tsp_instance)
        self.assertEqual(self.tsp_instance.matrix.shape, (6, 6))
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, base))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))


    def test_inplace_mutation_uniform(self):
        original_matrix = self.tsp_instance.matrix.copy()
        InplaceMutation('uniform', 20).mutate(self.tsp_instance)
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        # only two(upper triangle + lower triangle) elements should be changed
        self.assertEqual(differences, 2)  

class TestRandomSampling(unittest.TestCase):
    def test_fresh_instance_mutation_euclidean(self):
        builder = (
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("euclidean")
            .set_distribution("lognormal")
            .set_control(100)
            .set_dimensions(2)
        )
        
        # Create the original TSP instance
        original = builder.build()
        original_matrix = original.matrix.copy()
        
        # Apply "mutation" - which is actually a fresh instance generation and not really a mutation
        mutation = RandomSampling(builder)
        mutation.mutate(original)  # Modifies the instance in-place        

        assert original.tsp_type == "euclidean"
        assert not np.array_equal(original.matrix, original_matrix), "Matrix was not regenerated."
        assert np.allclose(original.matrix, original.matrix.T), "Matrix is not symmetric."
        assert np.all(np.diag(original.matrix) == np.inf), "Diagonal not set to infinity."
        

    def test_fresh_instance_mutation_asymmetric(self):
        builder = (
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(1.0)  # Sigma for lognormal
        )
        
        original = builder.build()
        original_matrix = original.matrix.copy()
        
        # Again "mutation" is a fresh instance generation and not really a mutation
        mutation = RandomSampling(builder)
        mutation.mutate(original)
        
        assert original.tsp_type == "asymmetric"
        assert not np.array_equal(original.matrix, original_matrix), "Matrix was not regenerated."
        # Validate asymmetric properties
        assert not np.allclose(original.matrix, original.matrix.T), "Matrix should not be symmetric."
        assert np.all(np.diag(original.matrix) == np.inf), "Diagonal not set to infinity."

# class TestRunMutation(unittest.TestCase):
#     # Test the mutation strategies
#     test_suite = unittest.TestSuite()
#     test_suite.addTest(unittest.makeSuite(TestATSPMutation))
#     test_suite.addTest(unittest.makeSuite(TestETSPMutation))
#     runner = unittest.TextTestRunner()
#     runner.run(test_suite)
    

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from core.generate_tsp import TSPBuilder, TSPInstance
from core.mutate_tsp import SwapMutation, ScrambleMutation, InplaceMutation, RandomSampling, get_mutation_strategy
from core.helpers import run_litals_algorithm
import itertools
from icecream import ic
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

    def test_asymmetric_tsp_lognormal(self):
        tsp = TSPBuilder().set_city_size(5).set_generation_type("asymmetric") \
                          .set_distribution("lognormal").set_control(100).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertFalse(np.allclose(tsp.matrix, tsp.matrix.T))

    def test_euclidean_tsp_lognormal(self):
        tsp = TSPBuilder().set_city_size(5).set_generation_type("euclidean") \
                          .set_distribution("lognormal").set_control(0.5).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertTrue(np.allclose(tsp.matrix, tsp.matrix.T))

    def test_euclidean_tsp_uniform(self):
        tsp = TSPBuilder().set_city_size(5).set_generation_type("euclidean") \
                          .set_distribution("uniform").set_control(0.5).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertTrue(np.allclose(tsp.matrix, tsp.matrix.T))
        mask = ~np.eye(tsp.matrix.shape[0], dtype=bool)
        assert np.all((tsp.matrix[mask] >= 0) & (tsp.matrix[mask] <= 100)), "Values out of bounds."

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
        # -- Test for multiple times to see if same elements repeatedly

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
        self.assertEqual(differences, 1)

    def test_inplace_mutation_lognormal(self):
        original_matrix = self.tsp_instance.matrix.copy()
        InplaceMutation('lognormal', 1.2).mutate(self.tsp_instance)
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        self.assertEqual(differences, 1)

class TestETSPMutation(unittest.TestCase):

    def setUp(self):
        np.random.seed(58)
        self.builder = TSPBuilder()
        self.tsp_instance = self.builder.set_city_size(6).set_generation_type("euclidean").set_distribution("uniform").set_control(20).build()
    
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
        tspinstance = InplaceMutation('uniform', 20)
        tspinstance.mutate(self.tsp_instance)
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        # only two(upper triangle + lower triangle) elements should be changed
        self.assertEqual(differences, 2)          

class TestRandomSampling(unittest.TestCase):
    def test_fresh_instance_mutation_euclidean(self):
        builder = (        
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("euclidean")
            .set_distribution("uniform") # lognormal produces values closer
            .set_control(500)
            .set_dimensions(100)
        )
        original = builder.build()
        original_matrix = original.matrix.copy()
        mutation = RandomSampling(builder)
        mutation.mutate(original)     

        assert original.tsp_type == "euclidean"
        assert not np.array_equal(original.matrix, original_matrix), "Matrix was not regenerated."
        assert np.allclose(original.matrix, original.matrix.T), "Matrix is not symmetric."
        assert np.all(np.diag(original.matrix) == np.inf), "Diagonal not set to infinity."
        n = original.matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        difference = np.sum(np.abs(original.matrix[mask] - original_matrix[mask]))
        assert difference > n, "Matrices are too identical."                

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
        # self.assertEqual(self.tsp_instance.matrix.shape, (5, 5))
        # Validate asymmetric properties
        assert not np.allclose(original.matrix, original.matrix.T), "Matrix should not be symmetric."
        assert np.all(np.diag(original.matrix) == np.inf), "Diagonal not set to infinity."

def brute_force_tsp_cost(matrix: np.ndarray) -> float:
    """
    Compute the true optimal tour cost of a full‐matrix TSP by brute force.
    Assumes square matrix with matrix[i,i] == np.inf.
    Uses city 0 as the fixed start/end.
    """
    n = matrix.shape[0]
    best = np.inf
    # permute the remaining cities
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm + (0,)
        cost = sum(matrix[tour[i], tour[i+1]] for i in range(n))
        if cost < best:
            best = cost
    return best

class TestSolvingTSPInstances(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(20)
        )
        self.tsp = builder.build()
        # brute force:
        self.opt0 = brute_force_tsp_cost(self.tsp.matrix)

    def test_original_instance(self):
        # check if Lital's algorithms can solve the original instance
        iter, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, self.opt0)

    def test_multiple_lital_are_same(self):
        # check if Lital's algorithm will produce the same iteration on the same tsp instance
        iter1, tour1, cost1, error1 = run_litals_algorithm(self.tsp.matrix)
        iter2, tour2, cost2, error2 = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error1)
        self.assertIsNone(error2)
        self.assertEqual(cost1, cost2)
        self.assertEqual(tour1, tour2)
        self.assertEqual(iter1, iter2)

class TestSolveSwapInstance(unittest.TestCase):
    def setUp(self):
        np.random.seed(89)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(20)
        )
        self.tsp = builder.build()
    
    def test_swap_and_solve(self):
        original_matrix = self.tsp.matrix.copy()
        # perform swap mutation
        SwapMutation().mutate(self.tsp)
        # check the property of the mutated tsp
        self.assertFalse(np.array_equal(self.tsp.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp.matrix))))
        # compute expected optimal cost for mutated instance
        expected_cost = brute_force_tsp_cost(self.tsp.matrix)
        iter_count, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected_cost)
        self.assertGreater(iter_count, 1)

class TestSolveScrambleInstance(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("asymmetric")
            .set_distribution("lognormal")
            .set_control(0.5)
        )
        self.tsp = builder.build()

    def test_scramble_and_solve(self):
        original_matrix = self.tsp.matrix.copy()
        # perform scramble mutation
        ScrambleMutation().mutate(self.tsp)
        self.assertFalse(np.array_equal(self.tsp.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp.matrix))))
        # compute expected optimal cost for mutated instance
        expected_cost = brute_force_tsp_cost(self.tsp.matrix)
        iter_count, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected_cost)
        self.assertGreater(iter_count, 1)

class TestInplaceMutation(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("euclidean")
            .set_distribution("uniform")
            .set_control(20)
        )
        self.tsp = builder.build()

    def test_inplace_mutation_and_solve(self):
        original_matrix = self.tsp.matrix.copy()
        # perform inplace mutation
        InplaceMutation('uniform', 20).mutate(self.tsp)
        self.assertFalse(np.array_equal(self.tsp.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp.matrix))))
        # compute expected optimal cost for mutated instance
        expected_cost = brute_force_tsp_cost(self.tsp.matrix)
        iter_count, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected_cost)
        self.assertGreater(iter_count, 1)
        
class TestRandomSamplingMutation(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("euclidean")
            .set_distribution("uniform")
            .set_control(20)
        )
        self.tsp = self.builder.build()

    def test_random_sampling_mutation_and_solve(self):
        original_matrix = self.tsp.matrix.copy()
        # perform random sampling mutation
        RandomSampling(self.builder).mutate(self.tsp)
        self.assertFalse(np.array_equal(self.tsp.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp.matrix))))
        # compute expected optimal cost for mutated instance
        expected_cost = brute_force_tsp_cost(self.tsp.matrix)
        iter_count, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected_cost)
        self.assertGreater(iter_count, 1)
# class TestRunMutation(unittest.TestCase):
#     # Test the mutation strategies
#     test_suite = unittest.TestSuite()
#     test_suite.addTest(unittest.makeSuite(TestATSPMutation))
#     test_suite.addTest(unittest.makeSuite(TestETSPMutation))
#     runner = unittest.TextTestRunner()
#     runner.run(test_suite)
    

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from core.generate_tsp import TSPBuilder, TSPInstance
from core.mutate_tsp import SwapMutation, ScrambleMutation, InplaceMutation, RandomSampling, get_mutation_strategy
from core.helpers import run_litals_algorithm
import itertools
from icecream import ic

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

class TestTSPGeneration(unittest.TestCase):
    """
    Tests for generating TSP instances with various configurations.
    """
    def test_asymmetric_tsp_uniform(self):
        """
        Verify asymmetric TSP with uniform distribution:
        - Matrix shape is (5, 5).
        - Diagonal entries are infinite.
        - Matrix is not symmetric.
        - Non-diagonal values lie within [0, control].
        """
        control = 100
        size = 5
        tsp = (
            TSPBuilder()
            .set_city_size(size)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(control)
            .build()
        )
        self.assertEqual(tsp.matrix.shape, (size, size))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertFalse(np.allclose(tsp.matrix, tsp.matrix.T))
        # Ensure values are within uniform distribution bounds (0-100)
        mask = ~np.eye(tsp.matrix.shape[0], dtype=bool)  # Exclude diagonal
        assert np.all((tsp.matrix[mask] >= 0) & (tsp.matrix[mask] <= control)), "Uniform values out of bounds."

    def test_asymmetric_tsp_lognormal(self):
        """
        Verify asymmetric TSP with lognormal distribution:
        - Matrix shape is (5, 5).
        - Diagonal entries are infinite.
        - Matrix is not symmetric.
        """
        tsp = TSPBuilder().set_city_size(5).set_generation_type("asymmetric") \
                          .set_distribution("lognormal").set_control(1.0).build()
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertFalse(np.allclose(tsp.matrix, tsp.matrix.T))

    def test_euclidean_tsp_lognormal(self):
        """
        Verify Euclidean TSP with lognormal distribution:
        - Matrix shape is (5, 5).
        - Diagonal entries are infinite.
        - Matrix is symmetric.
        """
        tsp = (
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("euclidean")
            .set_distribution("lognormal")
            .set_control(0.5)
            .build()
        )
        self.assertEqual(tsp.matrix.shape, (5, 5))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertTrue(np.allclose(tsp.matrix, tsp.matrix.T))

    def test_euclidean_tsp_uniform(self):
        """
        Verify Euclidean TSP with uniform distribution:
        - Matrix shape is (5, 5).
        - Diagonal entries are infinite.
        - Matrix is symmetric.
        - Distances are bounded by control.
        """
        control = 100
        size = 5
        tsp = (
            TSPBuilder()
            .set_city_size(size)
            .set_generation_type("euclidean")
            .set_distribution("uniform")
            .set_control(control)
            .build()
        )
        self.assertEqual(tsp.matrix.shape, (size, size))
        self.assertTrue(np.all(np.isinf(np.diag(tsp.matrix))))
        self.assertTrue(np.allclose(tsp.matrix, tsp.matrix.T))
        mask = ~np.eye(tsp.matrix.shape[0], dtype=bool)
        assert np.all((tsp.matrix[mask] >= 0) & (tsp.matrix[mask] <= 100)), "Values out of bounds."
        values = tsp.matrix[mask]
        self.assertTrue(np.all((values >= 0) & (values <= control)))

class TestATSPMutation(unittest.TestCase):
    """
    Tests for mutation strategies on asymmetric TSP instances.
    """
    def setUp(self):
        """
        Initialize a 4-city asymmetric TSP with a fixed seed.
        """
        np.random.seed(42)
        self.tsp_instance = TSPBuilder().set_city_size(4).set_generation_type("asymmetric").set_distribution("uniform").set_control(20).build()
    
    def test_swap_mutation(self):
        """
        SwapMutation should swap exactly two off-diagonal entries:
        - Matrix shape remains (4, 4).
        - Diagonal entries remain infinite.
        - Exactly two values are swapped.
        """
        original_matrix = self.tsp_instance.matrix.copy()
        SwapMutation().mutate(self.tsp_instance)
        self.assertEqual(self.tsp_instance.matrix.shape, (4, 4))
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        self.assertEqual(differences, 2) # only two elements should be swapped

    def test_scramble_mutation(self):
        """
        ScrambleMutation should randomize off-diagonal entries:
        - Matrix shape and infinite diagonal preserved.
        - Multiset of off-diagonal values remains unchanged.
        """
        original = self.tsp_instance.matrix.copy()
        ScrambleMutation().mutate(self.tsp_instance)
        mutated = self.tsp_instance.matrix
        self.assertEqual(mutated.shape, original.shape)
        self.assertFalse(np.array_equal(self.tsp_instance.matrix, original))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))
        # entires of mutate should be the same as the original
        mask = ~np.eye(4, dtype=bool)
        self.assertCountEqual(
            original[mask].tolist(),
            mutated[mask].tolist(),
            "Off-diagonal values should only be scrambled, but values have changed."
        )

    def test_inplace_mutation_uniform(self):
        """
        InplaceMutation should change exactly one distance entry:
        - Difference count is one.
        """
        original_matrix = self.tsp_instance.matrix.copy()
        InplaceMutation('uniform', 20).mutate(self.tsp_instance)
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        self.assertEqual(differences, 1)

class TestETSPMutation(unittest.TestCase):
    """
    Tests for mutation strategies on Euclidean TSP instances.
    """
    def setUp(self):
        """
        Initialize a 6-city Euclidean TSP with a fixed seed for reproducibility.
        """
        np.random.seed(58)
        self.builder = TSPBuilder()
        self.tsp_instance = self.builder.set_city_size(6).set_generation_type("euclidean").set_distribution("uniform").set_control(20).build()
    
    def test_swap_mutation(self):
        """
        SwapMutation on symmetric matrix should swap two distances symmetrically:
        - Four entries (two pairs) differ.
        - Symmetry and infinite diagonal preserved.
        """
        original_matrix = self.tsp_instance.matrix.copy()
        SwapMutation().mutate(self.tsp_instance)
        mutated = self.tsp_instance.matrix
        self.assertEqual(mutated.shape, original_matrix.shape)
        self.assertFalse(np.array_equal(mutated, original_matrix))
        self.assertTrue(np.all(np.isinf(np.diag(mutated))))
        differences = (self.tsp_instance.matrix != original_matrix).sum()
        # Differences include each swapped entry twice (i,j and j,i)
        self.assertEqual(differences, 4)
        self.assertTrue(np.allclose(mutated, mutated.T), "Matrix must remain symmetric.")

    def test_scramble_mutation(self):
        """
        ScrambleMutation should permute off-diagonal entries symmetrically:
        - Multiset preserved.
        - Symmetry and infinite diagonal preserved.
        """
        original = self.tsp_instance.matrix.copy()
        ScrambleMutation().mutate(self.tsp_instance)
        mutated = self.tsp_instance.matrix
        self.assertEqual(mutated.shape, original.shape)
        self.assertFalse(np.array_equal(mutated, original))
        self.assertTrue(np.all(np.isinf(np.diag(self.tsp_instance.matrix))))
        mask = ~np.eye(6, dtype=bool)
        self.assertCountEqual(
            original[mask].tolist(),
            mutated[mask].tolist(),
            "Scrambled matrix must preserve all distances."
        )
        self.assertTrue(np.allclose(mutated, mutated.T))

    def test_inplace_mutation_uniform(self):
        """
        InplaceMutation should change one distance symmetrically:
        - Two entries (i,j and j,i) differ.
        - Matrix remains symmetric.
        """
        original = self.tsp_instance.matrix.copy()
        InplaceMutation('uniform', 20).mutate(self.tsp_instance)
        mutated = self.tsp_instance.matrix
        differences = (mutated != original).sum()
        self.assertEqual(differences, 2)  
        self.assertTrue(np.allclose(mutated, mutated.T))
        

class TestRandomSampling(unittest.TestCase):
    """
    Tests for RandomSampling strategy regenerating fresh TSP instances.
    """
    def test_fresh_instance_mutation_euclidean(self):
        """
        RandomSampling on Euclidean builder should:
        - Regenerate matrix different from original.
        - Preserve symmetry and infinite diagonal.
        - Change total distance by more than city count.
        """
        builder = (        
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("euclidean")
            .set_distribution("uniform") # lognormal produces values closer
            .set_control(500)
            .set_dimensions(100)
        )
        tsp = builder.build()
        original_matrix = tsp.matrix.copy()
        mutation = RandomSampling(builder)
        mutation.mutate(tsp) 
        mutated = tsp.matrix
        n = mutated.shape[0]
        mask = ~np.eye(n, dtype=bool)    

        assert tsp.tsp_type == "euclidean"
        self.assertEqual(mutated.shape, original_matrix.shape)
        assert not np.array_equal(mutated, original_matrix), "Matrix was not regenerated."
        assert np.all(np.diag(mutated) == np.inf), "Diagonal not set to infinity."
        assert np.allclose(mutated, mutated.T), "Random sampled ETSP is not symmetric."
        difference = np.sum(np.abs(tsp.matrix[mask] - original_matrix[mask]))
        assert difference > n, "Matrices are too identical."                

    def test_fresh_instance_mutation_asymmetric(self):
        """
        RandomSampling on asymmetric builder should:
        - Regenerate matrix different from original.
        - Preserve infinite diagonal.
        - Matrix remains asymmetric.
        """
        builder = (
            TSPBuilder()
            .set_city_size(5)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(1.0)  # Sigma for lognormal
        )
        tsp = builder.build()
        original_matrix = tsp.matrix.copy()
        # "mutation" is a fresh instance generation and not really a mutation
        RandomSampling(builder).mutate(tsp)
        mutated = tsp.matrix
        self.assertEqual(mutated.shape, original_matrix.shape)
        assert tsp.tsp_type == "asymmetric"
        assert not np.array_equal(mutated, original_matrix), "Matrix was not regenerated."
        assert not np.allclose(mutated, mutated.T), "Matrix should not be symmetric."
        assert np.all(np.diag(mutated) == np.inf), "Diagonal not set to infinity."



class TestSolvingTSPInstances(unittest.TestCase):
    """
    Tests for run_litals_algorithm wrapper against brute-force solutions.
    """
    def setUp(self):
        """
        Initialize an asymmetric TSP with known optimal cost.
        """
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
        """
        run_litals_algorithm returns the optimal cost for the original instance without error.
        """
        iter, tour, cost, error = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, self.opt0)

    def test_multiple_lital_are_same(self):
        """
        Multiple run_litals_algorithm invocations on the same instance yield identical results.
        """
        # check if Lital's algorithm will produce the same iteration on the same tsp instance
        iter1, tour1, cost1, error1 = run_litals_algorithm(self.tsp.matrix)
        iter2, tour2, cost2, error2 = run_litals_algorithm(self.tsp.matrix)
        self.assertIsNone(error1)
        self.assertIsNone(error2)
        self.assertEqual(cost1, cost2)
        self.assertEqual(tour1, tour2)
        self.assertEqual(iter1, iter2)

class TestMutationSolveIntegration(unittest.TestCase):
    """
    Integration tests: apply mutations then solve with run_litals_algorithm.
    """

    def test_swap_and_solve(self):
        """
        Swap mutation followed by solver produces correct optimal cost.
        """
        np.random.seed(89)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("asymmetric")
            .set_distribution("uniform")
            .set_control(20)
        )
        instance = builder.build()
        original = instance.matrix.copy()
        SwapMutation().mutate(instance)
        self.assertTrue(np.all(np.isinf(np.diag(instance.matrix))))
        self.assertFalse(np.array_equal(instance.matrix, original))
        expected = brute_force_tsp_cost(instance.matrix)
        iter, _, cost, error = run_litals_algorithm(instance.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected)
        self.assertGreater(iter, 1)

    def test_scramble_and_solve(self):
        """
        Scramble mutation followed by solver produces correct optimal cost.
        """
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("asymmetric")
            .set_distribution("lognormal")
            .set_control(0.5)
        )
        instance = builder.build()
        original_matrix = instance.matrix.copy()
        ScrambleMutation().mutate(instance)
        mutated = instance.matrix
        self.assertFalse(np.array_equal(mutated, instance))
        self.assertTrue(np.all(np.isinf(np.diag(mutated))))
        self.assertEqual(original_matrix.shape, mutated.shape)
        expected = brute_force_tsp_cost(instance.matrix)
        iter, _, cost, error = run_litals_algorithm(instance.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected)
        self.assertGreater(iter, 1)

    def test_inplace_and_solve(self):
        """
        Inplace mutation followed by solver produces correct optimal cost.
        """
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("euclidean")
            .set_distribution("uniform")
            .set_control(20)
        )
        instance = builder.build()
        InplaceMutation('uniform', 20).mutate(instance)
        expected = brute_force_tsp_cost(instance.matrix)
        iter_count, _, cost, error = run_litals_algorithm(instance.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected)
        self.assertGreater(iter_count, 1)

    def test_random_sampling_and_solve(self):
        """
        RandomSampling followed by solver produces correct optimal cost.
        """
        np.random.seed(123)
        builder = (
            TSPBuilder()
            .set_city_size(6)
            .set_generation_type("euclidean")
            .set_distribution("uniform")
            .set_control(20)
        )
        instance = builder.build()
        RandomSampling(builder).mutate(instance)
        expected = brute_force_tsp_cost(instance.matrix)
        iter_count, _, cost, error = run_litals_algorithm(instance.matrix)
        self.assertIsNone(error)
        self.assertEqual(cost, expected)
        self.assertGreater(iter_count, 1)

if __name__ == '__main__':
    unittest.main()

import numpy as np
from core.generate_tsp import generate_tsp
from core.mutate_tsp import swap_mutation

def test_swap_mutation_diagonal_and_inf():
    """
    Test swap_mutation on both symmetric (Euclidean) and asymmetric TSP matrices.
    Ensures:
    - Diagonal entries remain np.inf after mutation.
    - No non-diagonal entries become np.inf.
    """
    # Test parameters
    n = 5  # Matrix size (small enough for testing, large enough for valid swaps)
    distributions = ['uniform', 'lognormal']
    
    for distribution in distributions:
        # Test symmetric (Euclidean) matrix
        symmetric_matrix = generate_tsp(
            city_size=n,
            generation_type="euclidean",
            distribution=distribution,
            control=100
        )
        mutated_symmetric = swap_mutation(symmetric_matrix)
        
        # Check diagonal is all inf
        assert np.all(np.diag(mutated_symmetric) == np.inf, \
            "Diagonal elements altered in symmetric matrix"
        
        # Check no inf in non-diagonal entries
        assert not np.any(mutated_symmetric[~np.eye(n, dtype=bool)] == np.inf), \
            "Non-diagonal inf introduced in symmetric matrix"
        
        # Test asymmetric matrix
        asymmetric_matrix = generate_tsp(
            city_size=n,
            generation_type="asymmetric",
            distribution=distribution,
            control=100
        )
        mutated_asymmetric = swap_mutation(asymmetric_matrix)
        
        # Check diagonal is all inf
        assert np.all(np.diag(mutated_asymmetric) == np.inf), \
            "Diagonal elements altered in asymmetric matrix"
        
        # Check no inf in non-diagonal entries
        assert not np.any(mutated_asymmetric[~np.eye(n, dtype=bool)] == np.inf), \
            "Non-diagonal inf introduced in asymmetric matrix"

test_swap_mutation_diagonal_and_inf()
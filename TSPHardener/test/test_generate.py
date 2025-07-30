import numpy as np

def triangle_inequality(matrix: np.ndarray) -> bool:
        """Test the triangle inequality for a symmetric distance matrix."""
        n = matrix.shape[0]
        # Iterate over all possible triplets (i, j, k) where i, j, k are distinct
        for i in range(n):
            for j in range(n):
                if i == j or matrix[i, j] == np.inf:
                    continue
                for k in range(n):
                    if k == i or k == j or matrix[i, k] == np.inf or matrix[j, k] == np.inf:
                        continue
                    # Check if the triangle inequality holds
                    if matrix[i, j] + matrix[j, k] < matrix[i, k]:
                        return False

        return True
import numpy as np

def generate_lognormal_matrix(rows, cols, mean=0, sigma=1):
    """
    Generate a matrix of lognormally distributed random numbers.

    Parameters:
    rows (int): Number of rows in the matrix.
    cols (int): Number of columns in the matrix.
    mean (float): Mean of the underlying normal distribution.
    sigma (float): Standard deviation of the underlying normal distribution.

    Returns:
    np.ndarray: A matrix of lognormally distributed random numbers.
    """
    normal_matrix = np.random.normal(mean, sigma, (rows, cols))
    lognormal_matrix = np.exp(normal_matrix)
    return lognormal_matrix

# Example usage
if __name__ == "__main__":
    rows = 5
    cols = 5
    matrix = generate_lognormal_matrix(rows, cols)
    print(matrix)
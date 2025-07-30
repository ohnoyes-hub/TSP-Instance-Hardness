import numpy as np
import logging
from scipy.spatial.distance import pdist, squareform
from icecream import ic

import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

LOGNORMAL_MEAN = 10

class TSPInstance:
    """
    A class representing a TSP instance with a distance matrix.
    """
    def __init__(self, matrix, tsp_type):
        self.matrix = matrix
        self.tsp_type = tsp_type

class TSPBuilder:
    """
    A builder class for generating TSP instances."""
    def __init__(self):
        self.city_size = None
        self.generation_type = None
        self.distribution = None
        self.control = None
        self.dimensions = 100  # Default dimension for Euclidean TSP

    def set_city_size(self, size):
        """
        Set the size of the city (number of cities) for the TSP instance.
        """
        if not isinstance(size, int) or size <= 2:
            raise ValueError("City size must be a positive integer.")
        self.city_size = size
        return self

    def set_generation_type(self, generation_type):
        """
        Set the type of TSP instance to generate (symmetric or asymmetric)."""
        assert generation_type in ["euclidean", "asymmetric"], "Invalid generation type"
        self.generation_type = generation_type
        return self

    def set_distribution(self, distribution):
        """
        Set the distribution to use for generating the TSP instance (uniform or lognormal)."""
        assert distribution in ["uniform", "lognormal"], "Invalid distribution type"
        self.distribution = distribution
        return self

    def set_control(self, control):
        """
        Set the control parameter for the distribution."""
        self.control = control
        return self

    def set_dimensions(self, dimensions):
        """
        Set the dimensions for the Euclidean TSP instance."""
        self.dimensions = dimensions
        return self

    def build(self):
        """
        Build the TSP instance based on the parameters set in the builder.
        """
        if None in [self.city_size, self.generation_type, self.distribution, self.control]:
            raise ValueError("One or more required parameters not set.")

        if self.generation_type == "euclidean":
            matrix = self._generate_euclidean_tsp()
            return TSPInstance(matrix, "euclidean")
        elif self.generation_type == "asymmetric":
            matrix = self._generate_asymmetric_tsp()
            return TSPInstance(matrix, "asymmetric")
        else:
            raise ValueError("Generation type not set or invalid")

    def _generate_asymmetric_tsp(self):
        """
        Generate a distance matrix for an asymmetric TSP instance based on the specified distribution and control parameter.
        The distances are integer values. The diagonal is set to infinity to indicate no self-loops.
        """
        if self.distribution == 'uniform':
            matrix = np.random.randint(0, self.control + 1, size=(self.city_size, self.city_size)).astype(float)
        else:  # lognormal
            matrix = np.around(np.random.lognormal(LOGNORMAL_MEAN, self.control, (self.city_size, self.city_size)))

        np.fill_diagonal(matrix, np.inf)
        return matrix

    def _generate_euclidean_tsp(self):
        """
        Generate a Euclidean distance matrix for the specified number of cities and dimensions.
        Points are generated based on the specified distribution and control parameter on a grid. The points are integer values.
        The grid is 100x100 unless specified otherwise. This is scaled to have a mean distance of 10.
        The distances are calculated using the Euclidean distance formula. The distance matrix is symmetric.
        The euclidean distances are rounded to the nearest integer.
        The diagonal is set to infinity to indicate no self-loops.
        """
        if self.distribution == 'uniform':
            scale = self.control / np.sqrt(self.dimensions)
            points = np.random.randint(0, int(scale) + 1, size=(self.city_size, self.dimensions))
        else:  # lognormal
            points = np.random.lognormal(LOGNORMAL_MEAN, self.control, (self.city_size, self.dimensions))
            scaling_factor = 10 / np.mean(np.linalg.norm(points, axis=1))
            points = np.around(points * scaling_factor).astype(int)

        distance_matrix = squareform(np.around(pdist(points)))
        np.fill_diagonal(distance_matrix, np.inf)
        return distance_matrix
    
# import pandas as pd
# ETSP = TSPBuilder().set_city_size(8).set_generation_type("euclidean").set_distribution("lognormal").set_control(1.0).build()
# df = pd.DataFrame(ETSP.matrix)
# df = df.replace(np.inf, r'\infy')
# latex = df.to_latex(index=False, header=False, caption="Euclidean TSP Instance", label="tab:euclidean_tsp")
# print(latex)

# ###############
# # Heatmap 
# ###############
# import matplotlib.pyplot as plt
# import numpy as np
# from generate_tsp import TSPBuilder

# # Configuration
# city_size = 20
# control_uniform = 30
# control_lognorm = 1.2
# plt.style.use('seaborn-v0_8-darkgrid')  # Modern and professional style

# # Generate uniform asymmetric instance
# builder_uniform = (
#     TSPBuilder()
#     .set_city_size(city_size)
#     .set_generation_type('asymmetric')
#     .set_distribution('uniform')
#     .set_control(control_uniform)
# )
# instance_uniform = builder_uniform.build()
# matrix_uniform = instance_uniform.matrix.copy()
# matrix_uniform[np.isinf(matrix_uniform)] = np.nan

# # Generate lognormal Euclidean instance
# builder_lognorm = (
#     TSPBuilder()
#     .set_city_size(city_size)
#     .set_generation_type('euclidean')
#     .set_distribution('lognormal')
#     .set_control(control_lognorm)
# )
# instance_lognorm = builder_lognorm.build()
# matrix_lognorm = instance_lognorm.matrix.copy()
# matrix_lognorm[np.isinf(matrix_lognorm)] = np.nan

# # Create figure with enhanced layout
# fig, axs = plt.subplots(1, 2, figsize=(14, 6), 
#                         gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.25})
# fig.suptitle(f'TSP Distance Matrices With {city_size} Cities', 
#              fontsize=16, fontweight='bold', y=0.98)

# # Uniform heatmap
# im1 = axs[0].imshow(matrix_uniform, cmap='viridis')
# axs[0].set_title(f'Asymmetric TSP\nUniform Distribution [0, {control_uniform}]', fontsize=13)
# axs[0].set_xlabel('To City', fontsize=10)
# axs[0].set_ylabel('From City', fontsize=10)
# cbar1 = fig.colorbar(im1, ax=axs[0], shrink=0.8)
# cbar1.set_label('Distance', fontsize=9)

# # Lognormal heatmap
# im2 = axs[1].imshow(matrix_lognorm, cmap='magma')
# axs[1].set_title(f'Euclidean TSP\nLognormal Distribution ($\mu=10$, $\sigma={control_lognorm}$)', fontsize=13)
# axs[1].set_xlabel('To City', fontsize=10)
# axs[1].set_ylabel('From City', fontsize=10)
# cbar2 = fig.colorbar(im2, ax=axs[1], shrink=0.8)
# cbar2.set_label('Distance', fontsize=9)

# # Add grid lines for better readab  ility
# for ax in axs:
#     ax.set_xticks(np.arange(-0.5, city_size, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, city_size, 1), minor=True)
#     ax.grid(which='minor', color='w', linestyle='-', linewidth=0.3)
#     ax.tick_params(which='minor', size=0)
#     ax.set_xticks(np.arange(0, city_size, 5))
#     ax.set_yticks(np.arange(0, city_size, 5))

# # Final polish
# plt.tight_layout(pad=3.0)
# plt.subplots_adjust(top=0.88)
# plt.savefig('tsp_heatmaps.png', dpi=300, bbox_inches='tight')
# plt.show()


# import numpy as np
# import logging
# from scipy.spatial.distance import pdist, squareform

# logger = logging.getLogger(__name__)

# LOGNORMAL_MEAN = 10

# def generate_asymmetric_tsp(n: int, distribution: str, control: float) -> np.ndarray:
#     """
#     Generate a random cost matrix of size n x n with values in the range [1, upper) for a asymmetric TSP.
    
#     Parameters:
#     ----------
#     n : int
#         The size of the square matrix.
#     distribution : str
#         The distribution to sample the costs from (either 'uniform' or 'lognormal').
#     control : float
#         The control parameter for the distribution.
#         - For 'uniform', this is the upper bound for the random values.
#         - For 'lognormal', this is the sigma parameter.
#     """
#     if distribution == 'uniform':
#         if control <= 0 or not isinstance(control, int):
#             raise ValueError("Control parameter must be a positive integer.")
#         matrix = np.random.randint(0, control + 1, size=(n, n)).astype(float)
#         matrix = _set_diagonal_to_inf(matrix)
#     elif distribution == 'lognormal':
#         matrix = np.around(np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=control, size=(n, n)))
#         matrix = _set_diagonal_to_inf(matrix)
#     else:
#         raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
#     return matrix

# def generate_euclidean_tsp(n: int, distribution: str, control: float, dimensions: int = 100) -> np.ndarray:
#     """
#     Generate a Euclidean distance matrix for n points in a specified number of dimensions.
    
#     Parameters:
#     ----------
#     n : int
#         The number of points.
#     distribution : str
#         The distribution to sample the points from (either 'uniform' or 'lognormal').
#     control : int
#         The control parameter for the distribution.
#         - For 'uniform', this is the upper bound for the random values.
#         - For 'lognormal', this is the sigma parameter.
#     dimensions : int
#         The dimensionality of the Cartesian plane (default is 10).
#     Returns:
#     -------
#     np.ndarray
#         A symmetric distance matrix of size n x n.
#     """
#     # Generate random coordinates for n points in the given dimension
#     if distribution == 'uniform':
#         # scale coordinates to cap at control parameter
#         scale = control / np.sqrt(dimensions)
#         points = np.random.randint(0, int(scale) + 1, size=(n, dimensions))
#     elif distribution == 'lognormal':
#         points = np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=control, size=(n, dimensions))
#         # scale coordinates to have a mean distance of 10
#         scaling_factor = 10 / np.mean(np.linalg.norm(points, axis=1))
#         points *= scaling_factor
#         points = np.around(points).astype(int)
#     else:
#         raise ValueError("Invalid distribution. Choose either 'uniform' or 'lognormal'.")
    
#     distances = pdist(points, metric='euclidean')
#     distance_matrix = squareform(np.around(distances))
#     distance_matrix = _set_diagonal_to_inf(distance_matrix)

#     return distance_matrix
#     # Calculate the pairwise Euclidean distance matrix
#     #distance_matrix = np.zeros((n, n), dtype=float)
    
#     # for i in range(n):
#     #     for j in range(i + 1, n):
#     #         distance = np.linalg.norm(points[i] - points[j])
#     #         distance = np.around(distance)  # Around to nearest integer
#     #         distance_matrix[i, j] = distance
#     #         distance_matrix[j, i] = distance
    
#     # np.fill_diagonal(distance_matrix, np.inf)  # Set diagonal to infinity for TSP
    

#     #return distance_matrix


# def generate_tsp(city_size, generation_type, distribution, control) -> np.ndarray:
#     """
#     Generate a TSP instance with the specified parameters.
    
#     Parameters:
#     ----------
#     city_size : int
#         The number of cities in the TSP instance.
#     generation_type : str
#         The type of TSP instance to generate (symmetric or asymmetric).
#     distribution : str
#         The distribution to use for generating the TSP instance (uniform or lognormal).
#     control : float
#         The control parameter for the distribution.
#         - For the uniform , this is the upper bound for cost values in the matrix.
#         - For the lognormal, this is the sigma parameter.
#     Returns:
#     -------
#     np.ndarray
#         The generated TSP instance.
#     """
#     if generation_type == "euclidean":
#         return generate_euclidean_tsp(city_size, distribution, control) # dimension of grid is 100x100 unless stated otherwise
#     elif generation_type == "asymmetric":
#         return generate_asymmetric_tsp(city_size, distribution, control)
#     else:
#         raise ValueError("Invalid generation type. Choose either 'euclidean' or 'asymmetric'.")

# def _set_diagonal_to_inf(matrix: np.ndarray) -> np.ndarray:
#     np.fill_diagonal(matrix, np.inf)
#     return matrix

# # from icecream import ic
# # # # values 
# # matrix = generate_tsp(4, "euclidean", "lognormal", 2.4)
# # ic(matrix)

# # ic(generate_tsp(4, "euclidean", "lognormal", 2.4))
# # ic(generate_tsp(4, "asymmetric", "lognormal", 0.2))
# # ic(generate_tsp(4, "euclidean", "uniform", 100))
# # ic(generate_tsp(4, "asymmetric", "uniform", 100))


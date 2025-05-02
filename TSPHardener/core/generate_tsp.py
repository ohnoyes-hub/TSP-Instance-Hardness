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
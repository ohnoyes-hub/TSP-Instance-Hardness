import numpy as np

class Euclidean_TSP:
    """
    Represents a Euclidean TSP problem where cities are represented by points in the plane.
    """ 
    def __init__(self, city_id = None, x = None, y = None):
        self.city_id = city_id
        self.x = x
        self.y = y
    
    def distance(self, other):
        """
        Returns the Euclidean distance between this city and another city.
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __repr__(self):
        return f"City {self.city_id} ({self.x}, {self.y})"
    
    def __str__(self):
        return f"City {self.city_id} ({self.x}, {self.y})"
    
def create_cost_matrix(cities):
    """
    Creates a cost matrix for a list of cities.
    """
    n = len(cities)
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                cost_matrix[i, j] = np.inf
            else:
                cost_matrix[i, j] = cities[i].distance(cities[j])
    
    return cost_matrix
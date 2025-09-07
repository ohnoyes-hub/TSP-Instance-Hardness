import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from icecream import ic

def plot_etsp(points, distance_matrix):
    plt.figure(figsize=(6, 6))
    for i, (x, y) in enumerate(points):
        plt.scatter(x, y, c='blue', s=100)
        # plt.text(x, y, str(i), fontsize=12, ha='right', )
    # Draw lines for all pairs (optional: only for selected pairs)
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'k--', alpha=0.2)
    plt.title(f'city size: {len(points)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()



def generate_euclidean_tsp(city_size=20, dimensions=2, control=5.0, distribution='uniform'):
    LOGNORMAL_MEAN = 1.0  # adjust as needed

    if distribution == 'uniform':
        scale = control / np.sqrt(dimensions)
        points = np.random.randint(0, int(scale) + 1, size=(city_size, dimensions))
    else:  # lognormal
        points = np.random.lognormal(LOGNORMAL_MEAN, control, (city_size, dimensions))
        scaling_factor = 10 / np.mean(np.linalg.norm(points, axis=1))
        points = np.around(points * scaling_factor).astype(int)

    distance_matrix = squareform(np.around(pdist(points)))
    np.fill_diagonal(distance_matrix, np.inf)

    # Mapping: each entry [i, j] corresponds to distance between points[i] and points[j]
    mapping = []
    for i in range(city_size):
        for j in range(city_size):
            if i != j:
                mapping.append({
                    'from': i,
                    'to': j,
                    'coords_from': points[i],
                    'coords_to': points[j],
                    'distance': distance_matrix[i, j]
                })
    return points, distance_matrix, mapping

# Example usage
points, distance_matrix, mapping = generate_euclidean_tsp(city_size=8, control=100)

print("City coordinates:")
print(points)
print("\nDistance matrix:")
print(distance_matrix)
print("\nMapping of matrix entries to point pairs:")
for m in mapping:
    print(f"distance_matrix[{m['from']}, {m['to']}] between {m['coords_from']} and {m['coords_to']} = {m['distance']}")

plot_etsp(points, distance_matrix)


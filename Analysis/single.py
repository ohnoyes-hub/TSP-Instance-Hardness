import json
import os


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import icecream as ic

# # Generate a 4-city TSP distance matrix
# n_cities = 4
# np.random.seed(42)  # Ensure reproducibility

# # Create a symmetric distance matrix with random values
# distance_matrix = np.random.randint(10, 100, size=(n_cities, n_cities))
# distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Symmetrize
# np.fill_diagonal(distance_matrix, np.inf)  # Set diagonal to infinity

# # Extract unique distances (excluding infinities)
# unique_distances = distance_matrix[np.triu_indices(n_cities, k=1)]
# unique_distances = unique_distances[~np.isinf(unique_distances)]  # Remove infinities

# # Plot the histogram
# plt.figure(figsize=(8, 6))
# sns.histplot(unique_distances, bins='auto', kde=True, color='blue', edgecolor='black')
# plt.title("Histogram of Distances in 4-City TSP")
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# plt.show()



path = 'CombinedData/uniform_asymmetric_wouter_size20/run1/result20_5_wouter.json' # uniform distribution [5, 10, .., 50]
#path = '../CombinedData/lognormal_asymmetric_wouter_size20/run1/result20_0.2_wouter.json' # uniform distribution [0.2, 0.4, ..., 5.0]

with open(path, 'r') as f:
    data = json.load(f)


count_iteration = len(data['results'].keys()) - 2


print(data.keys())
print(data['configuration'])
print(data['results'].keys())   
print(f"count_iteration {len(data['results'].keys()) - 2}") # -2 for initial_matrix and iteration_0
print(data['results']['iteration_1'].keys())
# print(data['results']['iteration_1']['iterations'])
print(data['results']['iteration_3']['hardest'])
print(f"Checks if this iterations is the hardest: {data['results']['iteration_4']['is_hardest']}")
# print(data['results']['initial_matrix'])
print(data['results']['iteration_1'].keys())
print(data['results']['iteration_1'].keys())
# print(np.array(data['results']['iteration_1']['matrix']))
# data['results']['iteration_2']['matrix']
# # ic(matrix1)

import numpy as np

def triangle_inequality_violations(cost_matrix):
    n = cost_matrix.shape[0]
    violations = 0
    severity = []
    total_triplets = n * (n - 1) * (n - 2) // 6

    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if i != k and j != k:
                        if cost_matrix[i, k] > cost_matrix[i, j] + cost_matrix[j, k]:
                            violations += 1
                            severity.append(cost_matrix[i, k] / (cost_matrix[i, j] + cost_matrix[j, k]))

    degree_of_violations = violations / total_triplets
    average_severity = np.mean(severity) if severity else 0

    return degree_of_violations, average_severity



import pandas as pd
import itertools

def generate_experiment_table():
    # Define parameters
    city_sizes = [20, 30]
    tsp_variants = ["Euclidean", "Asymmetric"]
    cost_distributions = ["Lognormal", "Uniform"]
    mutation_types = ["Scramble", "Inplace", "Swap"]

    # Generate all combinations
    combinations = list(itertools.product(city_sizes, tsp_variants, cost_distributions, mutation_types))

    # Create a DataFrame
    df = pd.DataFrame(combinations, columns=["City Size", "TSP Variant", "Cost Distribution", "Mutation Type"])

    # Add a Flag column with custom rules
    conditions = [
        (df["City Size"] == 20) & (df["TSP Variant"] == "Euclidean") & (df["Cost Distribution"] == "Lognormal"),
        (df["City Size"] == 20) & (df["TSP Variant"] == "Euclidean") & (df["Cost Distribution"] == "Uniform") & (df["Mutation Type"] == "Inplace"),
        (df["City Size"] == 20) & (df["TSP Variant"] == "Asymmetric") & (df["Cost Distribution"] == "Uniform") & (df["Mutation Type"] == "Scramble"),
        (df["City Size"] == 30) & (df["TSP Variant"] == "Asymmetric") & (df["Cost Distribution"] == "Lognormal") & (df["Mutation Type"] == "Scramble"),
        (df["City Size"] == 30) & (df["TSP Variant"] == "Asymmetric") & (df["Cost Distribution"] == "Uniform") & (df["Mutation Type"] == "Inplace"),
    ]

    # Define flag values for each condition
    flag_values = ["Red", "Green", "Green", "Green", "Green"]

    # Apply conditions to determine the Flag column
    df["Flag"] = "Red"  # Default flag
    for condition, flag in zip(conditions, flag_values):
        df.loc[condition, "Flag"] = flag

    # Print the DataFrame
    print(df)

    return df
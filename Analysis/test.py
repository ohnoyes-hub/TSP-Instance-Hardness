import pandas as pd
from itertools import product

def generate_experiment_combinations():
    # Define the parameters
    city_sizes = [20, 30]
    tsp_variants = ["Euclidean", "Asymmetric"]
    cost_distributions = ["Lognormal", "Uniform"]
    mutation_types = ["Scramble", "Inplace", "Swap"]

    # Generate all combinations using itertools.product
    combinations = list(product(city_sizes, tsp_variants, cost_distributions, mutation_types))

    # Convert combinations into a DataFrame
    df = pd.DataFrame(combinations, columns=["City Size", "TSP Variant", "Cost Distribution", "Mutation Type"])

    # Print the DataFrame
    print(df)

    return df

# Generate and display the DataFrame
df_combinations = generate_experiment_combinations()

# Flags(G, Y, R) for which congiration is complted by the experiment
df_combinations["Flag"] 


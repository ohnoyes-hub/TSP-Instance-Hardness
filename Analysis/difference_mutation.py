import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Base directory for JSON files
base_dir = "./CombinedData/"

# Helper function to extract mutation type from filenames
def extract_mutation(filename):
    return filename.split("_")[-1].split(".")[0]

# Helper function to decode "Infinity" strings in the matrix
def decode_matrix(matrix):
    # Replace "Infinity" strings with np.inf and return the NumPy array
    return np.array([[np.inf if val == "Infinity" else val for val in row] for row in matrix])

# Extract data for plotting
plot_data = []

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                mutation_type = extract_mutation(file)
                keys = sorted(data["results"].keys(), key=lambda x: int(x.split("_")[1]))  # Sort by generation
                
                # Calculate matrix differences
                for i in range(1, len(keys)):
                    prev_key = keys[i - 1]
                    curr_key = keys[i]
                    
                    # Decode matrices and calculate Frobenius norm
                    prev_matrix = decode_matrix(data["results"][prev_key]["matrix"])
                    curr_matrix = decode_matrix(data["results"][curr_key]["matrix"])
                    matrix_diff = np.linalg.norm(curr_matrix - prev_matrix, ord='fro')
                    
                    # Append data for plotting
                    iterations = data["results"][curr_key]["iterations"]
                    plot_data.append({
                        "Iteration": iterations,
                        "TSP Difference": matrix_diff,
                        "Mutation Type": mutation_type
                    })
            except Exception as e:
                print(f"Error processing file {file}: {e}")


# Convert to DataFrame
df = pd.DataFrame(plot_data)

# Generate scatter plot for matrix differences
plt.figure(figsize=(12, 6))
for mutation, group in df.groupby("Mutation Type"):
    plt.scatter(group["Iteration"], group["TSP Difference"], label=mutation, alpha=0.6)

plt.title("TSP Matrix Difference vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("TSP Difference (Frobenius Norm)")
plt.legend(title="Mutation Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./plots/matrix_difference_vs_iterations.png")
plt.show()

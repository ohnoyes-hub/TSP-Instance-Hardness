import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Base directory for JSON files
base_dir = "./CombinedData/"

# Helper function to extract mutation type from filenames
def extract_mutation(filename):
    # Assuming mutation is the last part of the filename before ".json"
    return filename.split("_")[-1].split(".")[0]

# Extract data for the scatter plot
scatter_data = []

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                mutation_type = extract_mutation(file)
                
                # Extract iteration counts
                for key in data["results"].keys():
                    if key.startswith("iteration_"):
                        generation = int(key.split("_")[1])
                        iteration_count = data["results"][key]["iterations"]
                        scatter_data.append({
                            "Generation": generation,
                            "Iterations": iteration_count,
                            "Mutation": mutation_type
                        })
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(scatter_data)

# Filter data: Only include generations < 6000
df_filtered = df[df["Generation"] < 6000]

# Generate scatter plot for all filtered data
plt.figure(figsize=(12, 6))
for mutation, group in df_filtered.groupby("Mutation"):
    plt.scatter(group["Generation"], group["Iterations"], label=mutation, alpha=0.6)

plt.title("Scatter Plot of Generations vs Iterations (Generations < 6000)")
plt.xlabel("Generation")
plt.ylabel("Iterations")
plt.legend(title="Mutation Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./plots/scatter_plot_filtered_by_generation-mutation.png")
plt.show()

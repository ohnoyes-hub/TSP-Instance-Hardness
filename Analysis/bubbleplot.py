import os
import json
import pandas as pd
import seaborn as sns
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

df_filtered = df[df["Iterations"] > 10000]

# Add density column by grouping and counting mutation occurrences
df_filtered['Density'] = df_filtered.groupby('Mutation')['Mutation'].transform('count')

# Generate Seaborn bubble plot
plt.figure(figsize=(14, 7))
sns.scatterplot(
    data=df_filtered,
    x="Generation", 
    y="Iterations", 
    hue="Mutation",
    size="Density",
    sizes=(100, 1000),  # Adjust bubble size range
    alpha=0.6,
    palette="viridis"
)

plt.title("Bubble Plot of Generations vs Iterations (Iteration > 10000)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Iterations", fontsize=14)
plt.legend(title="Mutation Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./plots/bubble_plot_generation_iteration.png")
plt.show()

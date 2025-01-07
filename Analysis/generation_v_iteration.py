import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Base directory for JSON files
base_dir = "./CombinedData"

# Helper function to extract configuration details
def extract_configuration(data):
    return (
        data["configuration"]["distribution"],
        data["configuration"]["generation_type"],
        data["configuration"]["mutation_type"],
        data["configuration"]["city_size"]
    )

# Extract data for the scatter plot
scatter_data = []

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                config = extract_configuration(data)
                config_label = "_".join(map(str, config))
                
                # Extract iteration counts
                for key in data["results"].keys():
                    if key.startswith("iteration_"):
                        generation = int(key.split("_")[1])
                        iteration_count = data["results"][key]["iterations"]
                        scatter_data.append({
                            "Generation": generation,
                            "Iterations": iteration_count,
                            "Configuration": config_label
                        })
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(scatter_data)

# Generate scatter plot for all iterations
plt.figure(figsize=(12, 6))
for config, group in df.groupby("Configuration"):
    plt.scatter(group["Generation"], group["Iterations"], label=config, alpha=0.6)

plt.title("Scatter Plot of Generations vs Iterations (All Data)")
plt.xlabel("Generation")
plt.ylabel("Iterations")
plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./plots/scatter_plot_all.png")
plt.show()

# Filter data for iterations > 10,000
df_filtered = df[df["Iterations"] > 10000]

# Generate scatter plot for iterations > 10,000
plt.figure(figsize=(12, 6))
for config, group in df_filtered.groupby("Configuration"):
    plt.scatter(group["Generation"], group["Iterations"], label=config, alpha=0.6)

plt.title("Scatter Plot of Generations vs Iterations (Iterations > 10,000)")
plt.xlabel("Generation")
plt.ylabel("Iterations")
plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./plots/scatter_plot_filtered.png")
plt.show()

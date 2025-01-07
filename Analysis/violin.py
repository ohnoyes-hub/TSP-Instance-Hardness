# Gives the total number of iterations for each configuration to show frequency or count_iteratons
# less count_iterations means more mutation were skipped due to try-catch. Likely, IndexError
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

base_dir = "./CombinedData/"

configurations = []
runs_data = {}

# Helper function to extract configuration details
def extract_configuration(data):
    return (
        data["configuration"]["distribution"],
        data["configuration"]["generation_type"],
        data["configuration"]["mutation_type"],
        data["configuration"]["city_size"]
    )


# Extract iterations data
boxplot_data = []

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
                        iteration_count = data["results"][key]["iterations"]
                        boxplot_data.append({"Configuration": config_label, "Iterations": iteration_count})
            except Exception as e:
                print(f"Error processing file {file}: {e}")


# Convert to DataFrame
df = pd.DataFrame(boxplot_data)

# Filter data: Only include iterations greater than 10,000
df_filtered = df[df["Iterations"] > 10000]

# Check if the filtered DataFrame is not empty
if not df_filtered.empty:
    # Generate the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_filtered, x="Configuration", y="Iterations", palette="muted")

    # Enhance plot aesthetics
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution of Lital's Iterations by Configuration (Iterations > 10,000)")
    plt.xlabel("Configuration")
    plt.ylabel("Iterations")

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("./plots/violin_plot_filtered.png")
    plt.show()
else:
    print("No data available for plotting after filtering.")

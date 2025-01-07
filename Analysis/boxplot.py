# Gives the total number of iterations for each configuration to show frequency or count_iteratons
# less count_iterations means more mutation were skipped due to try-catch. Likely, IndexError
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

base_dir = "./CombinedData"

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

# # Convert to DataFrame
# import pandas as pd
df = pd.DataFrame(boxplot_data)

# Create a boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x="Configuration", y="Iterations")
plt.xticks(rotation=90, ha='right')
plt.xlabel("Configuration")
plt.ylabel("Iterations")
plt.title("Distribution of Iterations Across Configurations")
plt.tight_layout()
plt.show()
#plt.savefig("./Analysis/iterations_boxplot.png")

# Calculate median iterations per configuration
# config_medians = df.groupby("Configuration")["Iterations"].median()

# # Split configurations into "low" and "high" groups based on median threshold
# threshold = config_medians.median()  # Use the median of medians as a threshold
# low_configs = config_medians[config_medians <= threshold].index
# high_configs = config_medians[config_medians > threshold].index

# # Separate data for the two groups
# df_low = df[df["Configuration"].isin(low_configs)]
# df_high = df[df["Configuration"].isin(high_configs)]

# Plot "low" iterations configurations
# plt.figure(figsize=(14, 8))
# sns.boxplot(data=df_low, x="Configuration", y="Lital's Iterations")
# plt.xticks(rotation=90, ha='right')
# plt.xlabel("Configuration (Low Iterations)")
# plt.ylabel("Iterations")
# plt.title("Distribution of Low Iterations Across Configurations")
# plt.tight_layout()
# plt.savefig("./Analysis/low_iterations_boxplot.png")

# Plot "high" iterations configurations
# plt.figure(figsize=(14, 8))
# sns.boxplot(data=df_high, x="Configuration", y="Lital's Iterations")
# plt.xticks(rotation=90, ha='right')
# plt.xlabel("Configuration (High Iterations)")
# plt.ylabel("Iterations")
# plt.title("Distribution of High Iterations Across Configurations")
# plt.tight_layout()
# plt.savefig("./Analysis/high_iterations_boxplot.png")
# plt.show()
# Gives the total number of iterations for each configuration to show frequency or count_iteratons
# less count_iterations means more mutation were skipped due to try-catch. Likely, IndexError
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

base_dir = "./Results"

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

# Iterate through the JSON files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            try:
                # Read JSON file
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                # Compute number of generations
                count_iteration = len(data["results"].keys()) - 2  # Subtract 2 for initial keys
                
                # Extract configuration as a tuple
                config = extract_configuration(data)
                
                # Determine run number from the file path
                run_number = os.path.basename(root).lower().replace("run", "")
                run_number = int(run_number) if run_number.isdigit() else 0
                
                # Initialize the configuration in the runs_data dictionary
                if config not in runs_data:
                    runs_data[config] = {}
                if run_number not in runs_data[config]:
                    runs_data[config][run_number] = 0
                
                # Accumulate generations for the specific run
                runs_data[config][run_number] += count_iteration
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Prepare data for the histogram
config_labels = ["_".join(map(str, config)) for config in runs_data.keys()]
runs = sorted({run for config in runs_data.values() for run in config.keys()})  # Unique sorted run numbers
data_matrix = np.zeros((len(config_labels), len(runs)))

# Populate the data matrix
for i, config in enumerate(runs_data.keys()):
    for j, run in enumerate(runs):
        data_matrix[i, j] = runs_data[config].get(run, 0)

# Create a horizontal stacked bar chart
plt.figure(figsize=(14, 8))
for j, run in enumerate(runs):
    plt.barh(config_labels, data_matrix[:, j], left=data_matrix[:, :j].sum(axis=1), label=f"Run {run}")

plt.yticks(rotation=0)
plt.ylabel("Configuration")
plt.xlabel("Total Generations")
plt.title("Total Generations Of Each Configuration and Run")
plt.legend(title="Run", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.show()
plt.savefig("./Analysis/coverage_horizontal.png")
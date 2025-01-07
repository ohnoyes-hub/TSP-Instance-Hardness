import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "./CombinedData"

# Helper function to extract configuration details
def extract_configuration(data):
    return (
        data["configuration"]["distribution"],
        data["configuration"]["generation_type"],
        data["configuration"]["mutation_type"],
        data["configuration"]["city_size"]
    )

stripplot_data = []

# Extract iterations and run data
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                config = extract_configuration(data)
                config_label = "_".join(map(str, config))
                
                # Extract run number from file path
                run_number = os.path.basename(root).lower().replace("run", "")
                run_number = int(run_number) if run_number.isdigit() else 0
                
                # Extract iteration counts
                for key in data["results"].keys():
                    if key.startswith("iteration_"):
                        iteration_count = data["results"][key]["iterations"]
                        stripplot_data.append({
                            "Configuration": config_label,
                            "Iterations": iteration_count,
                            "Run": run_number
                        })
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(stripplot_data)

# Create the stripplot
plt.figure(figsize=(14, 8))
sns.stripplot(
    data=df,
    x="Configuration",
    y="Iterations",
    hue="Run",
    dodge=True,  # Separate points for each run
    jitter=True,  # Spread points for better visibility
    palette="tab10"  # Distinct colors for runs
)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Configuration")
plt.ylabel("Iterations")
plt.title("Lital's Iterations by Configuration and Run")
plt.legend(title="Run", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./Analysis/stripplot.png")

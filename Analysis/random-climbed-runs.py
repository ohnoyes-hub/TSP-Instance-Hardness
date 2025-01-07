import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Directory containing the JSON files
base_dir = "./CombinedData/uniform_asymmetric_wouter_size30"

# List to hold the extracted data
data = []

# Loop through the files to extract control values, iterations, and runs
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json') and "wouter" in file:
            try:
                # Extract the control value from the filename
                control_value = int(file.split("_")[1])
                
                # Extract the run number from the directory name
                run_number = os.path.basename(root).lower().replace("run", "")
                run_number = int(run_number) if run_number.isdigit() else 0
                
                # Load the JSON file
                with open(os.path.join(root, file), 'r') as f:
                    json_data = json.load(f)
                
                # Extract iterations from the "results" section
                for key, value in json_data.get("results", {}).items():
                    if key.startswith("iteration_"):
                        iterations = value.get("iterations", None)
                        if iterations is not None:
                            data.append({
                                "Random Max": control_value,
                                "Iterations": iterations,
                                "Run": run_number
                            })
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# Create a DataFrame for plotting
df = pd.DataFrame(data)

# Exclude run 2
#df = df[df["Run"] != 2]

# Calculate the median iterations per run for each control value
median_df = df.groupby(["Random Max", "Run"]).median().reset_index()

# Set up the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Scatter plot with color indicating run number
scatter = sns.scatterplot(
    data=df,
    x="Random Max",
    y="Iterations",
    hue="Run",
    palette="tab10",
    alpha=0.7,
    s=50
)

# Line plot for the median iterations per run
line = sns.lineplot(
    data=median_df,
    x="Random Max",
    y="Iterations",
    hue="Run",
    palette="tab10",
    linewidth=2,
    legend=False  # Hide duplicate legend for lineplot
)

# Add custom legend entry for the median line
handles, labels = scatter.get_legend_handles_labels()
handles.append(plt.Line2D([], [], color='black', linewidth=2, label='Median Line'))
plt.legend(handles=handles, title="Run", fontsize=12, title_fontsize=13, loc="best")

# Plot customization
plt.title("Uniform Asymmetric Wouter Random Max vs Iterations", fontsize=16)
plt.xlabel("Random Max", fontsize=14)
plt.ylabel("Iterations", fontsize=14)
plt.tight_layout()
plt.savefig("./Analysis/random-climbed-runs.png")
plt.show()

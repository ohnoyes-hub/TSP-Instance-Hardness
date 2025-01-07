import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

# Directory containing the JSON files
base_dir = "./CombinedData/uniform_euclidean_scramble_size20"

# List to hold the extracted data
data = []

# Loop through directories and process only `run2`
for root, _, files in os.walk(base_dir):
    if "run1" in os.path.basename(root).lower():  # Check if it's run2
        for file in files:
            if file.endswith('.json') and "scramble" in file:  # Filter relevant files
                try:
                    # Load the JSON file
                    with open(os.path.join(root, file), 'r') as f:
                        json_data = json.load(f)
                    
                    # Extract generations, iterations, and hardest values
                    for iteration_key, values in json_data.get("results", {}).items():
                        if iteration_key.startswith("iteration_"):
                            generation = int(iteration_key.split("_")[1])  # Extract generation number
                            iterations = values.get("iterations", None)
                            hardest = values.get("hardest", None)
                            data.append({
                                "Generation": generation,
                                "Lital's iterations": iterations,
                            })
                except Exception as e:
                    print(f"Error processing file {file} in {root}: {e}")

# Ensure the data is ready
plot_data = pd.DataFrame(data)
plot_data.dropna(subset=["Lital's iterations"], inplace=True)

# Create a colormap for the gradient
norm = Normalize(vmin=plot_data["Lital's iterations"].min(), vmax=plot_data["Lital's iterations"].max())
colors = cm.viridis(norm(plot_data["Lital's iterations"]))

# Set theme for the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 9), facecolor='white')  # Widen the figure size

# Lineplot for Lital's iterations (connected line)
sns.lineplot(
    data=plot_data,
    x="Generation",
    y="Lital's iterations",
    label="Lital's Iterations",
    color="blue",
    linewidth=0.8,
    zorder=1
)

# Scatter plot with gradient colors for Lital's iterations
plt.scatter(
    plot_data["Generation"],
    plot_data["Lital's iterations"],
    color=colors,
    label="Lital's Iterations (Scatter)",
    s=15,
    zorder=2
)

# Add a colorbar to indicate gradient scale for the scatter plot
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

# Add title and labels
plt.title("Generations vs Lital's Iterations and Hardest Found (Cumulative)", size=16, fontweight='bold')
plt.xlabel("Generations", size=14, fontweight='bold')
plt.ylabel("Lital Iterations", size=14, fontweight='bold')

# Add legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)

# Save and display the plot
plt.savefig("./Analysis/uniform_euc_size20_litals_iterations_.png", bbox_inches='tight')
plt.show()
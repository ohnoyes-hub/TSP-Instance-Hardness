import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the file paths and output folder
file_paths = {
    #"uniform_asymmetric_wouter_size30": "./CombinedData/uniform_asymmetric_wouter_size30/run2/result30_5_wouter.json",
    "uniform_euclidean_wouter_size20": "./CombinedData/uniform_euclidean_wouter_size20/run2/result20_30_wouter.json"
}

output_folder = "./plots"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

def extract_data(file_path):
    data_points = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract configuration details for labeling
        config = (
            data["configuration"]["distribution"],
            data["configuration"]["generation_type"],
            data["configuration"]["mutation_type"],
            data["configuration"]["city_size"]
        )
        config_label = "_".join(map(str, config))

        # Extract iterations vs generation
        for key in data["results"].keys():
            if key.startswith("iteration_"):
                generation = int(key.split("_")[1])
                iteration_count = data["results"][key]["iterations"]
                data_points.append({
                    "Generation": generation,
                    "Iterations": iteration_count,
                    "Configuration": config_label
                })
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return data_points

# Plot using Seaborn
def plot_with_seaborn(df, config_name, output_folder):
    # Sort data by generation to ensure line connectivity
    df = df.sort_values(by="Generation")

    # Set up the plot aesthetics
    sns.set_style(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot the connected line using Seaborn's lineplot
    sns.lineplot(
        x="Generation", 
        y="Iterations", 
        data=df, 
        color="blue",  # Line color
        linewidth=2, 
        marker="o",   # Add markers for each point
        markersize=8, 
        linestyle="-"
    )

    # Annotate the highest iteration as generation progresses
    max_iteration = -1
    for i, row in df.iterrows():
        if row["Iterations"] > max_iteration:
            max_iteration = row["Iterations"]
            plt.annotate(
                f"{max_iteration}",
                (row["Generation"], row["Iterations"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white", alpha=0.7)
            )

    # Plot aesthetics
    plt.title(f"Random Climbed Generations vs Iterations ({config_name})")
    plt.xlabel("Generation")
    plt.ylabel("Iterations")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, f"{config_name}_connected.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

# Process each file and save the data + plots
for config_name, file_path in file_paths.items():
    # Extract data
    data = extract_data(file_path)
    if not data:
        print(f"No data extracted for {config_name}. Skipping...")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the data to a CSV file
    csv_path = os.path.join(output_folder, f"{config_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # Generate and save the plot
    plot_with_seaborn(df, config_name, output_folder)

print("Processing complete!")
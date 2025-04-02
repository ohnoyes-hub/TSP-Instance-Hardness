import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to extract configuration details
def extract_configuration(data):
    return (
        data["configuration"]["distribution"],
        data["configuration"]["generation_type"],
        data["configuration"]["mutation_type"],
        data["configuration"]["city_size"]
    )

# Helper function to extract control parameter from file name
def extract_control_parameter(file_name):
    try:
        parts = file_name.split("_")
        for part in parts:
            if part.replace(".", "", 1).isdigit():
                return float(part)
    except Exception as e:
        print(f"Error extracting control parameter from {file_name}: {e}")
    return None

# Specify base directory and configurations to process
base_dir = "./CombinedData"
#target_configurations = TODO

data = []

# Iterate through specified configurations
for config in target_configurations:
    config_dir = os.path.join(base_dir, config)
    if os.path.exists(config_dir):
        for run_folder in os.listdir(config_dir):
            run_path = os.path.join(config_dir, run_folder)
            if os.path.isdir(run_path):
                for file in os.listdir(run_path):
                    if file.endswith('.json'):
                        try:
                            file_path = os.path.join(run_path, file)
                            with open(file_path, 'r') as f:
                                json_data = json.load(f)
                            
                            control_param = extract_control_parameter(file)
                            for iteration_key, values in json_data.get("results", {}).items():
                                if iteration_key.startswith("iteration_"):
                                    generation = int(iteration_key.split("_")[1])
                                    hardest = values.get("hardest", None)
                                    data.append({
                                        "Configuration": config,
                                        "ControlParameter": control_param,
                                        "Generation": generation,
                                        "Hardest": hardest,
                                        "Run": run_folder
                                    })
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

# Create output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Filter data and plot for each configuration and control parameter
for config in target_configurations:
    config_data = df[df['Configuration'] == config]
    for control_param, group_data in config_data.groupby('ControlParameter'):
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=group_data,
            x='Generation',
            y='Hardest',
            hue='Run',
            markers=True,
            dashes=False,
            palette="tab10"
        )
        
        # Set plot title and labels
        title = f"{config} (Control Parameter: {control_param}) Hardest Values per Generation"
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Hardest')
        plt.legend(title='Run', loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        safe_config = config.replace(" ", "_").replace("/", "_")
        plot_filename = os.path.join(output_dir, f"{safe_config}_control_{control_param}_hardest_values.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to avoid overlap in the loop

        print(f"Plot saved: {plot_filename}")

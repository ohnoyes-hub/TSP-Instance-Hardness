import json
import os
import re

# File extracts the hardest matrices from JSON files and saves them to new JSON 
# This will be used to create the local optima network and train the convolutional neural network
input_dir = "../Data"
output_dir = "../CleanData"
os.makedirs(output_dir, exist_ok=True)

# Function to extract run number from file path
def extract_run_number(file_path):
    match = re.search(r'_run(\d+)', file_path)
    return int(match.group(1)) if match else None

# Function to extract configuration name from file path (including floating point values)
def extract_configuration_name(file_path):
    match = re.search(r'result(\d+_\d+\.\d+)_', file_path)
    return match.group(1) if match else "unknown_configuration"

# Dictionary to hold aggregated data for each configuration
aggregated_data = {}

# Function to process a JSON file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Extract configuration and iterate over results
        configuration = data.get("configuration", {})
        results = data.get("results", {})
        run_number = extract_run_number(file_path)
        configuration_name = extract_configuration_name(file_path)
        
        # Extract additional configuration details
        mutation_type = configuration.get("mutation_type", "unknown_mutation")
        generation_type = configuration.get("generation_type", "unknown_generation")
        distribution = configuration.get("distribution", "unknown_distribution")
        
        # Create a descriptive configuration name
        descriptive_config_name = f"{configuration_name}_{distribution}_{generation_type}_{mutation_type}"
        
        # Initialize lists to store all iterations and hardest values for the current run
        all_iterations = []
        hardest_iterations = []
        hardest_matrices = []
        
        for key, value in results.items():
            if isinstance(value, dict):
                all_iterations.append(value.get("iterations"))
                hardest_iterations.append(value.get("hardest"))
                
                # If is_hardest is true, extract the additional information
                if value.get("is_hardest"):
                    hardest_matrices.append({
                        "iteration": key,
                        "iterations": value.get("iterations"),
                        "hardest": value.get("hardest"),
                        "matrix": value.get("matrix")
                    })
        
        # Add current run data to aggregated data for the configuration
        if descriptive_config_name not in aggregated_data:
            aggregated_data[descriptive_config_name] = {
                "configuration": configuration,
                "runs": []
            }
        
        aggregated_data[descriptive_config_name]["runs"].append({
            "run_number": run_number,
            "iterations": all_iterations,
            "hardest": hardest_iterations,
            "hardest_matrices": hardest_matrices
        })

# Iterate through all JSON files in the input directory
for root, _, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".json"):
            file_path = os.path.join(root, file_name)
            process_file(file_path)

# TODO: Only lognormal_asymetric files are being processed all the rest are not being processed correctly

# Save aggregated data for each configuration to a separate summary file
for configuration_name, data in aggregated_data.items():
    summary_file_name = f"{configuration_name}_summary.json"
    summary_file_path = os.path.join(output_dir, summary_file_name)
    with open(summary_file_path, 'w') as summary_file:
        json.dump(data, summary_file)

print("Extraction completed.")

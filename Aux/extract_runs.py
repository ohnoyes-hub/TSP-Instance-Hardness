import json
import os
import re

# File extracts the runs from the JSON files in the Data directory
# This file will be used to replicate the hillclimber runs plot of Wouter's paper
input_dir = "../Data"
output_dir = "../CleanData"
os.makedirs(output_dir, exist_ok=True)


def extract_run_number(file_path):
    match = re.search(r'_run(\d+)', file_path)
    return int(match.group(1)) if match else None

# Initialize aggregated data
aggregated_data = {
    "runs": []
}

# Process a JSON file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Extract configuration and iterate over results
        configuration = data.get("configuration", {})
        results = data.get("results", {})
        run_number = extract_run_number(file_path)
        
        # Initialize lists to store all iterations and hardest values for the current run
        all_iterations = []
        hardest_iterations = []
        # hardest_matrices = []
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Append iteration and hardest to lists
                all_iterations.append(value.get("iterations"))
                hardest_iterations.append(value.get("hardest"))
                
                # # If is_hardest is true, extract the additional information
                # if value.get("is_hardest"):
                #     hardest_matrices.append({
                #         "iteration": key,
                #         "iterations": value.get("iterations"),
                #         "hardest": value.get("hardest"),
                #         "matrix": value.get("matrix")
                #     })
        
        # Add current run data to aggregated data
        aggregated_data["runs"].append({
            "configuration": configuration,
            "run_number": run_number,
            "iterations": all_iterations,
            "hardest": hardest_iterations,
            # "hardest_matrices": hardest_matrices
        })

# Iterate through all JSON files in the input directory
for root, _, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".json"):
            file_path = os.path.join(root, file_name)
            process_file(file_path)

# Save aggregated data to a single summary file
summary_file_path = os.path.join(output_dir, "run_summary.json")
with open(summary_file_path, 'w') as summary_file:
    json.dump(aggregated_data, summary_file)

print("Extraction completed.")

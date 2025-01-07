import os
import json

# Path to the folder containing JSON files
base_dir = "../CombinedData"
subfolders = [
    "lognormal_asymmetric_scramble_size20",
    "lognormal_asymmetric_scramble_size30",
    "uniform_asymmetric_scramble_size20",
    "uniform_asymmetric_wouter_size30", 
    "uniform_euclidean_wouter_size20"
]

# List to store all keys from JSON files
all_keys = []
file_count = 0
error_count = 0

# Traverse subfolders and analyze JSON files
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    if not os.path.exists(subfolder_path):
        print(f"Subfolder not found: {subfolder_path}")
        continue

    # Iterate over run folders
    for run_folder in os.listdir(subfolder_path):
        run_folder_path = os.path.join(subfolder_path, run_folder)
        if not os.path.isdir(run_folder_path):
            print(f"Skipping non-directory: {run_folder_path}")
            continue

        # Iterate over JSON files in each run folder
        for filename in os.listdir(run_folder_path):
            if filename.endswith(".json"):
                file_count += 1
                file_path = os.path.join(run_folder_path, filename)
                try:
                    with open(file_path, "r") as file:
                        data = json.load(file)
                        # Collect keys
                        all_keys.append(set(data.keys()))
                except json.JSONDecodeError:
                    error_count += 1
                    print(f"Error decoding JSON in file: {file_path}")
                except Exception as e:
                    error_count += 1
                    print(f"Unexpected error in file: {file_path} -> {e}")

unique_keys = set.union(*all_keys) if all_keys else set()
print("\n--- Summary ---")
print(f"Total files processed: {file_count}")
print(f"Files with errors: {error_count}")
print(f"Unique keys across all files: {unique_keys}")

# Error decoding JSON in file: ../CombinedData/lognormal_asymmetric_scramble_size20/run5/result20_3.8_scramble.json
''' Print Output
Skipping non-directory: ../CombinedData/lognormal_asymmetric_scramble_size20/log
Error decoding JSON in file: ../CombinedData/lognormal_asymmetric_scramble_size20/run5/result20_3.8_scramble.json
Skipping non-directory: ../CombinedData/uniform_euclidean_wouter_size20/log

--- Summary ---
Total files processed: 246
Files with errors: 1
Unique keys across all files: {'configuration', 'time', 'results'}
'''
import os
import json
from utils.json_utils import custom_decoder
from icecream import ic
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def validate_json_structure(file_path, data):
    """Validate the structure and content of the experiment JSON data."""
    errors = []
    warnings = []

    # Top-level keys check
    required_top_keys = {'configuration', 'time', 'results'}
    missing_top_keys = required_top_keys - data.keys()
    if missing_top_keys:
        errors.append(f"Missing top-level keys: {missing_top_keys}")

    # Check 'results' structure
    if 'results' in data:
        results = data['results']
        required_results_keys = {'hard_instances', 'last_matrix'}
        missing_results_keys = required_results_keys - results.keys()
        if missing_results_keys:
            errors.append(f"Missing 'results' keys: {missing_results_keys}")
        else:
            # Validate hard_instances entries
            hard_instances = results['hard_instances']
            for key, instance in hard_instances.items():
                if not key.startswith('iteration_'):
                    warnings.append(f"Invalid key in hard_instances: {key}")
                required_instance_keys = {'iterations', 'hardest', 'matrix', 'optimal_cost'}
                missing_instance_keys = required_instance_keys - instance.keys()
                if missing_instance_keys:
                    errors.append(f"Instance {key} missing keys: {missing_instance_keys}")

            # Validate last_matrix structure
            last_matrix = results['last_matrix']
            if not isinstance(last_matrix, list) or not all(isinstance(row, list) for row in last_matrix):
                errors.append("'last_matrix' must be a 2D list")

    # Check configuration content
    if 'configuration' in data:
        config = data['configuration']
        required_config_keys = {'mutation_type', 'generation_type', 'distribution'}
        missing_config_keys = required_config_keys - config.keys()
        if missing_config_keys:
            warnings.append(f"Configuration missing keys: {missing_config_keys}")

    # Check time is numeric
    if 'time' in data and not isinstance(data['time'], (int, float)):
        errors.append("'time' must be numeric")

    return errors, warnings

def load_json(file_path):
    """Load and validate a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f, object_hook=custom_decoder)
            errors, warnings = validate_json_structure(file_path, data)
            return data, errors, warnings
    except json.JSONDecodeError as e:
        return None, [f"JSON decode error: {str(e)}"], []
    except Exception as e:
        return None, [f"Unexpected error: {str(e)}"], []

def main():
    os.makedirs('./plot', exist_ok=True)
    # Paths to check - adjust according to your data location
    base_dirs = [
        #"../Results",
        "./Continuation"
    ]

    total_files = 0
    error_files = 0
    warning_files = 0
    detailed_issues = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            total_files += 1
            data, errors, warnings = load_json(file_path)

            # Collect issues regardless of errors
            if errors or warnings:
                entry = {
                    "file": file_path,
                    "errors": errors,
                    "warnings": warnings
                }
                detailed_issues.append(entry)
                if errors: error_files += 1
                if warnings: warning_files += 1

            # Generate plot if data is valid
            if data is not None and not errors:
                hard_instances = data['results']['hard_instances']
                plot_data = []
                
                for key, instance in hard_instances.items():
                    if not key.startswith('iteration_'):
                        continue
                    try:
                        iteration_num = int(key.split('_')[-1])
                    except ValueError:
                        continue
                    
                    # Extract values and ensure they are numeric
                    iterations_val = instance.get('iterations')
                    hardest_val = instance.get('hardest')
                    if not isinstance(iterations_val, (int, float)) or not isinstance(hardest_val, (int, float)):
                        continue
                    
                    plot_data.append({
                        'iteration': iteration_num,
                        'iterations': iterations_val,
                        'hardest': hardest_val
                    })
                
                if plot_data:
                    df = pd.DataFrame(plot_data)
                    df_melted = df.melt(id_vars='iteration', var_name='metric', value_name='value')
                    
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=df_melted, x='iteration', y='value', hue='metric', marker='o')
                    plt.title(f"Iteration Metrics: {os.path.basename(file_path)}")
                    plt.xlabel("Iteration Number")
                    plt.ylabel("Value")
                    
                    # Save plot
                    plot_name = os.path.basename(file_path).replace('.json', '.png')
                    plot_path = os.path.join('plot', plot_name)
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()    
                    
    # Print summary
    ic("=== Validation Summary ===")
    ic(total_files)
    ic(error_files)
    ic(warning_files)

    # Print detailed issues
    if detailed_issues:
        ic("=== Detailed Issues ===")
        for issue in detailed_issues:
            ic(issue['file'])
            if issue['errors']:
                ic("Errors:")
                for err in issue['errors']:
                    ic(err)
            if issue['warnings']:
                ic("  Warnings:")
                for warn in issue['warnings']:
                    ic(warn)

if __name__ == "__main__":
    main()
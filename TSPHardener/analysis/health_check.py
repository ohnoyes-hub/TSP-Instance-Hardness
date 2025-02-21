import os
import json
from utils.json_utils import custom_decoder
from icecream import ic

def validate_json_structure(file_path, data):
    """Validate the structure and content of the experiment JSON data."""
    errors = []
    warnings = []

    required_top_keys = {'configuration', 'time', 'results'}
    missing_top_keys = required_top_keys - data.keys()
    if missing_top_keys:
        errors.append(f"Missing top-level keys: {missing_top_keys}")

    # 'results' structure
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
                required_instance_keys = {'iterations', 'hardest', 'matrix'}
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

def main():
    base_dirs = [
        #"../Results",
        "../Continuation"
    ]

    total_files = 0
    error_files = 0
    warning_files = 0
    detailed_issues = []

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            ic(f"Directory not found: {base_dir}")
            continue

        for file in glob(f"{path}/**/*.json", recursive=True):


        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith(".json"):
                    total_files += 1
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f, object_hook=custom_decoder)
                            errors, warnings = validate_json_structure(file_path, data)
                            
                            if errors or warnings:
                                entry = {
                                    "file": file_path,
                                    "errors": errors,
                                    "warnings": warnings
                                }
                                detailed_issues.append(entry)
                                if errors: error_files += 1
                                if warnings: warning_files += 1

                    except json.JSONDecodeError as e:
                        error_files += 1
                        detailed_issues.append({
                            "file": file_path,
                            "errors": [f"JSON decode error: {str(e)}"],
                            "warnings": []
                        })
                    except Exception as e:
                        error_files += 1
                        detailed_issues.append({
                            "file": file_path,
                            "errors": [f"Unexpected error: {str(e)}"],
                            "warnings": []
                        })

    ic("=== Validation Summary ===")
    ic(total_files)
    ic(error_files)
    ic(warning_files)

    if detailed_issues:
        ic("=== Detailed Issues ===")
        for issue in detailed_issues:
            ic(issue['file'])
            if issue['errors']:
                ic("    Errors")
                for err in issue['errors']:
                    ic(err)
            if issue['warnings']:
                ic("  Warnings:")
                for warn in issue['warnings']:
                    ic(warn)

if __name__ == "__main__":
    main()
import os
import json
from utils.json_utils import custom_decoder
from icecream import ic
import glob

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
        # Use glob to recursively find all JSON files
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            total_files += 1
            data, errors, warnings = load_json(file_path)

            if errors or warnings:
                entry = {
                    "file": file_path,
                    "errors": errors,
                    "warnings": warnings
                }
                detailed_issues.append(entry)
                if errors: error_files += 1
                if warnings: warning_files += 1

    # Print summary
    print(f"\n=== Validation Summary ===")
    print(f"Total files checked: {total_files}")
    print(f"Files with errors: {error_files}")
    print(f"Files with warnings: {warning_files}")

    # Print detailed issues
    if detailed_issues:
        print("\n=== Detailed Issues ===")
        for issue in detailed_issues:
            print(f"\nFile: {issue['file']}")
            if issue['errors']:
                print("  Errors:")
                for err in issue['errors']:
                    print(f"  - {err}")
            if issue['warnings']:
                print("  Warnings:")
                for warn in issue['warnings']:
                    print(f"  - {warn}")

if __name__ == "__main__":
    main()

# with open("Continuation/uniform_asymmetric/city19_range17_wouter.json", 'r') as f:
#     data = json.load(f, object_hook=custom_decoder)

# ic(data.keys())
# ic(data['configuration'].keys())
# ic(data['results'].keys())
# ic(data['time'])
# ic(data['results']['hard_instances'].keys())
# ic(data['results']['hard_instances']['iteration_1'].keys())
# ic(data['results']['last_matrix'])
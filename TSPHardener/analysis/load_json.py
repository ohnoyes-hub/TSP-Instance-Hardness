import json
from utils.json_utils import custom_decoder
import os
import glob
from icecream import ic
import pandas as pd
from typing import Tuple, Dict, List

def validate_json_structure(data):
    """
    Validate the structure and content of the experiment JSON data.
    Parameters:
    ----------
    file_path : str
        The path to the JSON file being validated.
    data : dict
        The parsed JSON data.
    Returns:
    -------
    errors : list
        A list of error messages.
    warnings : list
        A list of warning messages.
    """
    errors = []
    warnings = []

    # Keys check
    required_top_keys = {'configuration', 'time', 'results'}
    missing_top_keys = required_top_keys - data.keys()
    if missing_top_keys:
        errors.append(f"Missing top-level keys: {missing_top_keys}")

    # Check 'results' structure
    if 'results' in data:
        results = data['results']
        required_results_keys = {'hard_instances', 'last_matrix', 'all_iterations'}
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
            
            # Validate all_iterations structure
            all_iterations = results.get('all_iterations', [])
            if not isinstance(all_iterations, list):
                errors.append("'all_iterations' must be a list of iteration integers")

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
    """
    Load and validate a JSON file.
    Parameters:
    ----------
    file_path : str
        The path to the JSON file.
    Returns:
    -------
    data : dict
        The parsed JSON data
    errors : list
        A list of error messages
    warnings : list
        A list of warning messages
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f, object_hook=custom_decoder)
    except json.JSONDecodeError as e:
        return None, [f"JSON decode error: {str(e)}"], []
    except Exception as e:
        return None, [f"Unexpected error: {str(e)}"], []
    
    errors, warnings = validate_json_structure(data)
    return data, errors, warnings

# def load_experiment_data(base_path: str = '.') -> Tuple[pd.DataFrame, List[str]]:
#     """
#     Load all experiment data from Results and Continuation folders into a DataFrame.
    
#     Parameters:
#     -----------
#     base_path : str, optional
#         Base directory path (default is current directory)
    
#     Returns:
#     --------
#     pd.DataFrame
#         Combined data with columns: ['city_size', 'range', 'mutation_type',
#         'generation_type', 'distribution', 'iteration']
#     List[str]
#         List of error messages for problematic files
#     """
#     # Collect files from both directories
#     pattern = os.path.join(base_path, '{Results,Continuation}', '**', 'city*_range*.json')
#     files = glob.glob(pattern, recursive=True)
    
#     data = []
#     errors = []
    
#     for file_path in files:
#         try:
#             # Extract parameters from filename
#             filename = os.path.basename(file_path)
#             city_part, range_part = filename.split('_')[:2]
#             city_size = int(city_part.replace('city', ''))
#             control_range = int(range_part.replace('range', '').split('.')[0])
            
#             # Load and validate JSON
#             json_data, file_errors, warnings = load_json(file_path)
            
#             if file_errors:
#                 errors.append(f"{filename}: {', '.join(file_errors)}")
#                 continue
                
#             # Extract configuration and iterations
#             config = json_data.get('configuration', {})
#             iterations = json_data.get('results', {}).get('all_iterations', [])
            
#             # Skip files with no iterations
#             if not iterations:
#                 errors.append(f"{filename}: No iterations found")
#                 continue
                
#             # Add each iteration as a row
#             for iteration in iterations:
#                 data.append({
#                     'city_size': city_size,
#                     'range': control_range,
#                     'mutation_type': config.get('mutation_type'),
#                     'generation_type': config.get('generation_type'),
#                     'distribution': config.get('distribution'),
#                     'iteration': iteration
#                 })
                
#         except Exception as e:
#             errors.append(f"{filename}: {str(e)}")
#             continue
    
#     return pd.DataFrame(data), errors

# # Load data from current directory
# df, errors = load_experiment_data()

# # Print summary
# print(f"Encountered {len(errors)} errors:\n", "\n".join(errors[:3]))

# # Quick stats
# print("\nConfiguration performance:")
# print(df.groupby(['distribution'])['iteration'].describe())
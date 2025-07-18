import json
from utils.json_utils import custom_decoder
import os
import glob
from icecream import ic
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, List

def validate_json_structure(data) -> Tuple[List, List]:
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
        required_config_keys = {'mutation_type', 'generation_type', 'distribution', 'city_size', 'range'}
        missing_config_keys = required_config_keys - config.keys()
        if missing_config_keys:
            warnings.append(f"Configuration missing keys: {missing_config_keys}")

    # Check time is numeric
    if 'time' in data and not isinstance(data['time'], (int, float)):
        errors.append("'time' must be numeric")

    return errors, warnings

def load_json(file_path) -> Tuple[Dict, List, List]:
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

        # Rename 'wouter' mutation type to 'inplace'
        if data and 'configuration' in data:
            config = data['configuration']
            if config.get('mutation_type') == 'wouter':
                config['mutation_type'] = 'inplace'

    except json.JSONDecodeError as e:
        return None, [f"JSON decode error: {str(e)}"], []
    except Exception as e:
        return None, [f"Unexpected error: {str(e)}"], []
    
    errors, warnings = validate_json_structure(data)
    return data, errors, warnings

def load_full():
    """
    Load all JSON files in the Continuation and Results directories.
    Returns:
    -------
    list
        A list of dictionaries containing validate JSON data.
    """
    base_dirs = [
        #"./Continuation",
        "./Results"
    ]

    all_data = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, warnings = load_json(file_path)
            if data is None or errors:
                ic("Error", file_path, errors)
                continue

            all_data.append(data)
    
    return all_data

def load_all_hard_instances() -> pd.DataFrame:
    """
    Load all hardest instances from JSON files along with their configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing each hardest instance's data merged with its configuration.
    """
    base_dirs = [
        "./Continuation", 
        "./Results"
    ]
    all_instances = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if data is None or errors:
                continue  # Skip invalid files
            
            config = data.get('configuration', {})
            hard_instances = data.get('results', {}).get('hard_instances', {})

            for key, instance in hard_instances.items():
                if not key.startswith('iteration_'):
                    continue
                
                try:
                    # generation number
                    iteration_num = int(key.split('_')[-1])
                except ValueError:
                    continue
                
                # Validate numerical values
                iterations_val = instance.get('iterations')
                hardest_val = instance.get('hardest')
                optimal_cost = instance.get('optimal_cost')


                if not isinstance(iterations_val, (int, float)) or not isinstance(hardest_val, (int, float)) or not isinstance(optimal_cost, (int, float)):
                    continue

                # Merge instance data with configuration
                entry = {
                    'generation': iteration_num,
                    'iterations': iterations_val,
                    'hardest_value': hardest_val,
                    'optimal_cost': optimal_cost,
                    "matrix" : instance.get('matrix'),
                    **config
                }
                all_instances.append(entry)
    
    return pd.DataFrame(all_instances)

def load_all_iteration():
    """
    Load all iterations from JSON files.
    """
    all_data = load_full()
    grouped_data = defaultdict(list)

    for entry in all_data:
        generation_type = entry.get('configuration', {}).get('generation_type', 'unknown')
        iterations = entry.get('results', {}).get('all_iterations', [])
        grouped_data[generation_type].extend(iterations)
    
    return grouped_data

def load_lon_data() -> Tuple[Dict, defaultdict]:
    """
    Loads all LON data (local_optima and transitions) from JSON files.
    """
    all_data = load_full()
    global_local_optima = {}
    global_transitions = defaultdict(list)

    for data in all_data:
        results = data.get('results', {})

        local_optima = {}
        for key_str, value in results.get("local_optima", {}).items():
            try:
                key = int(key_str)  # Convert JSON string keys to integers
                local_optima[key] = value
            except ValueError:
                continue  # Skip invalid keys
        
        # Process transitions (convert keys/dests to integers)
        transitions = results.get("transitions", {})
        for src_str, dests in transitions.items():
            try:
                src = int(src_str)  # Convert source key to integer
                valid_dests = [int(d) for d in dests if isinstance(d, (int, str))]
                global_transitions[src].extend(valid_dests)
            except ValueError:
                continue  # Skip invalid keys

        # Merge local_optima
        global_local_optima.update(local_optima)

    return global_local_optima, global_transitions


import os
import json
from utils.json_utils import custom_decoder
from icecream import ic
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        required_results_keys = {'hard_instances', 'last_matrix'} #, 'all_iterations'}
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


import json
import os
import glob
from icecream import ic
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, List
import re
import numpy as np


def custom_decoder(obj) -> Dict:
    """
    Custom decoder that converts "Infinity" to np.inf in nested structures.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v == "Infinity":
                obj[k] = np.inf
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value == "Infinity":
                obj[key] = np.inf
            elif isinstance(value, dict):
                obj[key] = custom_decoder(value)
            elif isinstance(value, list):
                obj[key] = custom_decoder(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if value == "Infinity":
                obj[i] = np.inf
            elif isinstance(value, dict):
                obj[i] = custom_decoder(value)
            elif isinstance(value, list):
                obj[i] = custom_decoder(value)
    return obj

def fill_missing_config(data, file_path):
    """
    Fill missing configuration values by parsing the file path using regex.
    """
    if not data or 'configuration' not in data:
        return

    config = data['configuration']
    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)

    # Regex pattern to extract parameters from directory and filename
    dir_pattern = re.compile(r"^(?P<distribution>\w+)_(?P<generation_type>\w+)$")
    file_pattern = re.compile(
        r"^city(?P<city_size>\d+)_range(?P<range>\d+\.?\d*)_(?P<mutation_type>\w+)\.json$"  # Allow floats
    )

    # Extract distribution and generation_type from parent directory
    if len(parts) >= 2:
        parent_dir = parts[-2]
        dir_match = dir_pattern.match(parent_dir)
        if dir_match:
            config.setdefault('distribution', dir_match.group("distribution"))
            config.setdefault('generation_type', dir_match.group("generation_type"))
    

    # Extract city_size, range, mutation_type from filename
    filename = os.path.basename(file_path)
    file_match = file_pattern.match(filename)
    if file_match:
        city_size = file_match.group("city_size")
        range_str = file_match.group("range")
        mutation_type = file_match.group("mutation_type")
        
        # Set city_size as integer
        config.setdefault('city_size', int(city_size))
        
        # Set mutation_type
        config.setdefault('mutation_type', mutation_type)
        
        # Parse range based on distribution
        distribution = config.get('distribution')
        try:
            if distribution == 'lognormal':
                config.setdefault('range', float(range_str))
            elif distribution == 'uniform':
                # Ensure range is integer (even if filename has .0)
                config.setdefault('range', int(float(range_str)))
            else:
                # Fallback: try float first, then int
                config.setdefault('range', float(range_str) if '.' in range_str else int(range_str))
        except ValueError:
            pass


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
        required_results_keys = {'initial_matrix', 'hard_instances', 'last_matrix', 'all_iterations', 'local_optima', 'transitions'}
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
        else:
            # Validate range type based on distribution
            distribution = config['distribution']
            range_val = config['range']
            if distribution == 'lognormal' and not isinstance(range_val, float):
                errors.append("Range must be a float for lognormal distribution")
            elif distribution == 'uniform' and not isinstance(range_val, int):
                errors.append("Range must be an integer for uniform distribution")
                
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
            

            # change mutation type naming convention
            if config.get('mutation_type') == 'wouter':
                config['mutation_type'] = 'inplace'

            # Fill missing configuration values
            fill_missing_config(data, file_path)

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
        "./Continuation",
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
            if warnings:
                ic("Warning", file_path, warnings)

            all_data.append(data)
    
    ic("Loaded", len(all_data), "files")
    
    return all_data

def load_all_hard_instances() -> pd.DataFrame:
    """
    Load all hardest instances from JSON files along with their configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing each hardest instance's data merged with its configuration.
    """
    base_dirs = [
        "../data/Continuation", 
        "../data/Results"
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
                    'file_path': file_path,
                    **config
                }
                all_instances.append(entry)
    
    ic("Loaded", len(all_instances), "hard instances")

    return pd.DataFrame(all_instances)

def load_initial_and_hard_instances():
    """
    Load both initial and hardest instances from experiment JSONs.
    Returns DataFrame with an extra column 'instance_type' = 'initial' or 'hardest'
    """
    all_instances = []
    base_dirs = ["../data/Continuation", "../data/Results"]

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if data is None or errors:
                continue
            config = data.get('configuration', {})
            results = data.get('results', {})
            # Initial instance
            if 'initial_matrix' in results:
                all_iterations = results.get('all_iterations', [])
                # Use the first iteration value if it exists, else fallback to 0
                initial_iteration = all_iterations[0] if (isinstance(all_iterations, list) and len(all_iterations) > 0) else 0
                entry = {
                    "instance_type": "initial",
                    "matrix": results['initial_matrix'],
                    "optimal_cost": results.get('optimal_cost', None),  # If available
                    "iteration": initial_iteration,
                    "generation": 0,
                    **config
                }
                all_instances.append(entry)

            # Hardest instances
            hard_instances = results.get('hard_instances', {})
            for key, instance in hard_instances.items():
                try:
                    iteration_num = int(key.split('_')[-1])
                except Exception:
                    continue
                entry = {
                    "instance_type": "hardest",
                    "matrix": instance.get('matrix'),
                    "optimal_cost": instance.get('optimal_cost', None),
                    "iteration": instance.get('iterations', None),
                    "generation": iteration_num,
                    **config
                }
                all_instances.append(entry)
    print("Loaded", len(all_instances), "initial and hardest instances")
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

    ic("Loaded", len(grouped_data), "generation types")
    
    return grouped_data

def load_phase_transition_iterations() -> pd.DataFrame:
    """
    Load all iterations from Phase Transition experiments (all_iterations field).
    """
    base_dirs = [
        # "../data/Phase-Trans-Merged", # From Phase Transition experiments : Final Thesis
        "../data/ResearchProject" # Research project phase 1
    ]
    all_entries = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if data is None or errors:
                continue  # Skip invalid files
            
            config = data.get('configuration', {})
            iterations = data.get('results', {}).get('all_iterations', [])
            
            for iteration_val in iterations:
                entry = {
                    'iteration': iteration_val,
                    'file_path': file_path,
                    **config
                }
                all_entries.append(entry)
    
    ic("Loaded", len(all_entries), "phase transition iterations")
    # print(all_entries)
    return pd.DataFrame(all_entries)

def load_hill_climb_iterations() -> pd.DataFrame:
    """
    Load all iterations from Hill Climbing experiments (all_iterations field) as well as its.
    """
    base_dirs = [
        "../data/Results",
        "../data/Continuation"
    ]
    all_entries = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)

        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            if data is None or errors:
                continue  # Skip invalid files
            
            config = data.get('configuration', {})
            iterations = data.get('results', {}).get('all_iterations', [])
            
            for iteration_val in iterations:
                entry = {
                    'iteration': iteration_val,
                    'file_path': file_path,
                    **config
                }
                all_entries.append(entry)
    
    ic("Loaded", len(all_entries), "hill climbed Lital iterations")
    return pd.DataFrame(all_entries)

def load_lon_data() -> Tuple[Dict, defaultdict]:
    """
    Loads all LON data (local_optima and transitions) from JSON files.
    """
    all_data = load_full()
    global_local_optima = {}
    global_transitions = defaultdict(set)

    for data in all_data:
        results = data.get('results', {})

        # Process local optima
        local_optima = {}
        for key_str, value in results.get("local_optima", {}).items():
            try:
                key = int(key_str)  # Convert JSON string keys to integers
                local_optima[key] = value
            except ValueError:
                ic("Invalid key", key_str)
                continue  # Skip invalid keys
        
        # Process transitions (convert keys/dests to integers)
        transitions = results.get("transitions", {})
        for src_str, dests in transitions.items():
            try:
                src = int(src_str)  # Convert source key to integer
                #valid_dests = [int(d) for d in dests if isinstance(d, (int, str))]
                valid_dests = {int(d) for d in dests if isinstance(d, (int, str))}
                global_transitions[src].update(valid_dests)
            except ValueError:
                ic("Invalid key", src)
                continue  # Skip invalid keys

        # Merge local optima and transitions
        global_local_optima.update(local_optima)

    # Ensure only valid optima in transitions
    for src in list(global_transitions.keys()):
        global_transitions[src] = {d for d in global_transitions[src] if d in local_optima}

    # convert transitions back to list for JSON compatibility
    final_transitions = {k: list(v) for k, v in global_transitions.items()}

    ic("Loaded", len(global_local_optima), "local optima and", len(global_transitions), "transitions")

    return global_local_optima, final_transitions

def fix_iteration_data(file_path, metric="sum"):
    """Convert 'iterations' list to a scalar metric."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    local_optima = data.get("results", {}).get("local_optima", {})
    for key, value in local_optima.items():
        iterations_list = value.get("iterations", [])
        if isinstance(iterations_list, list) and len(iterations_list) > 0:
            if metric == "sum":
                value["iterations"] = sum(iterations_list)
            elif metric == "avg":
                value["iterations"] = sum(iterations_list) / len(iterations_list)
            elif metric == "max":
                value["iterations"] = max(iterations_list)
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def filter_transitions(lon_data, cost_threshold=0.01):
    """
    Filter out transitions that are trivial (e.g., self-loops) or have similar costs."""
    cleaned_transitions = defaultdict(set)
    for source, targets in lon_data["filtered_transitions"].items():
        source_cost = lon_data["local_optima"].get(source, {}).get("cost", None)
        
        for target in targets:
            target_cost = lon_data["local_optima"].get(target, {}).get("cost", None)
            
            if source == target:
                continue  # Remove self-loops
            
            if source_cost is not None and target_cost is not None:
                if abs(source_cost - target_cost) < cost_threshold:
                    continue  # Remove trivial transitions
            
            cleaned_transitions[source].add(target)
    
    lon_data["filtered_transitions"] = {k: list(v) for k, v in cleaned_transitions.items()}
    return lon_data

def load_all_matrices() -> pd.DataFrame:
    """
    Load ALL generated matrices from phase transition experiments, along with their configuration and Lital iteration
    """
    base_dirs= [
        "../data/ResearchProject/Research-Project-Continuation",
        "../data/ResearchProject/Research-Project-Results"
    ]

    rows = []

    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        for file_path in glob.glob(pattern, recursive=True):
            data, errors, _ = load_json(file_path)
            if data is None or errors:
                continue

            config = data.get('configuration', {})
            results = data.get('results', {}) or {}
            all_mats = results.get('all_matrices')

            if all_mats is None:
                # Fall back: capture initial/hardest as a minimal baseline if needed.
                # Comment out if you only want runs that actually have all_matrices.
                init_mat = results.get('initial_matrix')
                if isinstance(init_mat, list):
                    rows.append({
                        'generation': 0,
                        'iteration': None,
                        'matrix': init_mat,
                        **config
                    })
                # Also capture last_matrix (optional)
                last_mat = results.get('last_matrix')
                if isinstance(last_mat, list):
                    rows.append({
                        'generation': -1,
                        'iteration': None,
                        'matrix': last_mat,
                        **config
                    })
                continue

            # Optional: map generation -> iteration value if you stored all_iterations
            all_iterations = results.get('all_iterations', [])
            def iteration_for(gen_idx):
                try:
                    return all_iterations[gen_idx] if isinstance(all_iterations, list) else None
                except Exception:
                    return None

            # Case A: all_matrices is a list
            if isinstance(all_mats, list):
                for gen_idx, entry in enumerate(all_mats):
                    if isinstance(entry, list):
                        # raw matrix
                        rows.append({
                            'generation': gen_idx,
                            'iteration': iteration_for(gen_idx),
                            'matrix': entry,
                            **config
                        })
                    elif isinstance(entry, dict):
                        # dict with potential fields
                        mat = entry.get('matrix', entry.get('mat', entry.get('M')))
                        if isinstance(mat, list):
                            rows.append({
                                'generation': gen_idx,
                                'iteration': entry.get('iterations', iteration_for(gen_idx)),
                                'matrix': mat,
                                **config
                            })

            # Case B: all_matrices is a dict (e.g., {"iteration_0": {...}, ...})
            elif isinstance(all_mats, dict):
                for key, entry in all_mats.items():
                    # try to parse generation index from key like "iteration_12"
                    gen_idx = None
                    if isinstance(key, str):
                        m = re.search(r'(\d+)', key)
                        if m:
                            gen_idx = int(m.group(1))
                    # fallback if no digits in key; push None which you can filter later
                    if isinstance(entry, list):
                        # raw matrix
                        rows.append({
                            'generation': gen_idx if gen_idx is not None else -2,
                            'iteration': iteration_for(gen_idx) if gen_idx is not None else None,
                            'matrix': entry,
                            **config
                        })
                    elif isinstance(entry, dict):
                        mat = entry.get('matrix', entry.get('mat', entry.get('M')))
                        if isinstance(mat, list):
                            rows.append({
                                'generation': gen_idx if gen_idx is not None else -2,
                                'iteration': entry.get('iterations', iteration_for(gen_idx) if gen_idx is not None else None),
                                'matrix': mat,
                                **config
                            })

    df = pd.DataFrame(rows)

    # (Optional) ensure dtypes you care about
    if 'generation' in df.columns:
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce').astype('Int64')

    # (Optional) convert matrix lists to numpy arrays for easier numeric analysis
    # Comment out if you prefer to keep Python lists
    try:
        df['matrix'] = df['matrix'].apply(lambda m: np.array(m, dtype=float) if isinstance(m, list) else m)
    except Exception:
        pass

    print("Loaded", len(df), "matrices from all experiments")
    return df
import json
import numpy as np
import os
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def custom_decoder(obj):
    """
    Custom decoder that converts "Infinity" to np.inf in nested structures.
    """
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

def custom_encoder(obj):
    """
    Custom encoder that turns np types into numpy types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif obj is np.inf:
        return "Infinity"
    return obj

def load_partial(cont_file):    
    """
    Loads partial data from a continuation file.
    Returns (hardest, matrix) so we can resume:
      - hardest = the largest #iterations found so far
      - matrix  = last_matrix (the matrix to mutate next iteration)
    
    The file structure has:
    {
      "time": <float>,
      "configuration": {...},
      "results": {
          "hard_instances": {
              "iteration_0": {...},  # new hardest at iteration_0
              "iteration_150": {...},  # next hardest
               ...
          },
          "last_matrix": [ ... ]  # the last hard matrix
      },
      "local_optima": {...},
    }
    """
    with open(cont_file, "r") as f:
        data = json.load(f, object_hook=custom_decoder)

    results = data["results"]
    if "hard_instances" not in results:
        # No stored info => fallback
        raise ValueError("No hard_instances in partial file!")
    if "last_matrix" not in results:
        raise ValueError("No last_matrix in partial file!")
    
    # find the largest iteration key in "hard_instances" to get the hardest
    # they look like "iteration_0", "iteration_150", ...
    # Ensure that continuation works on the hardest TSP instance
    max_iter = -1
    max_hard = 0
    for k, v in results["hard_instances"].items():
        # v is { "iterations": <int>, "hardest": <int>, ... }
        iter_idx = int(k.split("_")[1])
        if v["hardest"] > max_hard:
            max_hard = v["hardest"]
        if iter_idx > max_iter:
            max_iter = iter_idx

    hardest = max_hard
    matrix = np.array(results["last_matrix"])

    # Local optima and transitions
    results = data["results"]
    local_optima = results.get("local_optima", {})
    transitions = results.get("transitions", defaultdict(list))

    return hardest, matrix #, local_optima, transitions


def save_partial(configuration, results, citysize, rang, time_spent,
                  distribution, tsp_type, mutation_strategy, is_final=False):    
    """
    Saves partial (or final) JSON:
      - if not final => into 'Continuation'
      - if final => into 'Results' (and removes old continuation file).
    
    results has the structure:
      {
        "hard_instances": {
            "iteration_X": {
                "iterations": <Lital's iter>,
                "hardest": <int>,
                "matrix": <2D list>,
                ...
            },
            ...
        },
        "last_matrix": <the *current* matrix as a 2D list>,
        "all_iterations": <list of all Lital's iter>
      }
    """
    base_name = f"city{citysize}_range{rang}_{mutation_strategy}.json"
    folder = f"Results/{distribution}_{tsp_type}" if is_final else f"Continuation/{distribution}_{tsp_type}"
    
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    full_path = os.path.join(folder, base_name)

    # Merge into existing file if it exists
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            existing_data = json.load(f, object_hook=custom_decoder)
        existing_data["time"] += time_spent

        # Merge `all_iterations`
        existing_iterations = existing_data["results"].get("all_iterations", [])
        new_iterations = results.get("all_iterations", [])
        existing_data["results"]["all_iterations"] = existing_iterations + new_iterations  # Append new values

        # Update `last_matrix` and `hard_instances`
        existing_data["results"]["last_matrix"] = results["last_matrix"]
        existing_data["results"]["hard_instances"].update(results["hard_instances"])

        # Update `initial_matrix` if it's not already there
        if "initial_matrix" not in existing_data["results"]:
            existing_data["results"]["initial_matrix"] = results["initial_matrix"]

        # Update `local_optima` to be included in Continuation and Results
        existing_local_optima = existing_data["results"].get("local_optima", {})
        new_local_optima = results.get("local_optima", {})
        existing_data["results"]["local_optima"] = {**existing_local_optima, **new_local_optima}
        # Update `transitions` to be included in Continuation and Results
        existing_trans = existing_data["results"].get("transitions", defaultdict(list))
        existing_trans = defaultdict(list, existing_trans)
        new_trans = results.get("transitions", defaultdict(list))
        new_trans = defaultdict(list, new_trans)  # Ensure it's a defaultdict
        for src, dests in new_trans.items():
            existing_trans[src].extend(dests)
        # convert back to regular dict before saving
        existing_data["results"]["transitions"] = dict(existing_trans)
    else:
        existing_data = {
            "time": time_spent,
            "configuration": configuration,
            "results": results
        }
        

    # Write
    with open(full_path, "w") as f:
        json.dump(existing_data, f, indent=2, default=custom_encoder)
    
    #If final => remove from Continuation
    if is_final:
        logger.info(f"Saved final results to: {full_path}")
        cont_path = os.path.join("Continuation", f"{distribution}_{tsp_type}", base_name)
        if os.path.exists(cont_path):
            os.remove(cont_path)
            logger.info(f"Removed continuation file {cont_path}")
    else:
        logger.info(f"Saved partial results to: {full_path}")

def load_full_results(cont_file):
    """Loads entire results from a continuation file."""
    with open(cont_file, "r") as f:
        data = json.load(f, object_hook=custom_decoder)
    return data.get("results", {})
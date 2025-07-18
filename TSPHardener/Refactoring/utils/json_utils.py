import json
import numpy as np
import os
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

def custom_decoder(obj):
    """
    Custom decoder that converts "Infinity" to np.inf in nested structures.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = custom_decoder(v)
    elif isinstance(obj, list):
        return [custom_decoder(x) for x in obj]
    elif obj == "Infinity":
        return np.inf
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
    if "hard_instances" not in results or "last_matrix" not in results:
        raise ValueError("Missing required fields in partial file.")
    # Determine hardest value
    max_hard = max(v["hardest"] for v in results["hard_instances"].values())
    matrix = np.array(results["last_matrix"])
    local_optima = results.get("local_optima", {})
    transitions = results.get("transitions", defaultdict(list))
    return max_hard, matrix, local_optima, transitions


def save_partial(configuration, results, citysize, rang, time_spent,
                 distribution, tsp_type, mutation_strategy, is_final=False):  
    """
    Saves partial or final TSP experiment data.
    Merges with existing data if present, retains all fields.
    
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
    folder = Path("Results") / f"{distribution}_{tsp_type}" if is_final \
             else Path("Continuation") / f"{distribution}_{tsp_type}"
    folder.mkdir(parents=True, exist_ok=True)
    full_path = folder / base_name

    # Prepare new payload
    new_data = {
        "time": time_spent,
        "configuration": configuration,
        "results": results
    }

    # Merge if existing
    if full_path.exists():
        with open(full_path, "r") as f:
            existing_data = json.load(f, object_hook=custom_decoder)
        existing_data["time"] += time_spent
        existing_res = existing_data["results"]
        new_res = results

        existing_res["all_iterations"] = existing_res.get("all_iterations", []) + new_res.get("all_iterations", [])
        existing_res["last_matrix"] = new_res["last_matrix"]
        existing_res["hard_instances"].update(new_res["hard_instances"])
        if "initial_matrix" not in existing_res:
            existing_res["initial_matrix"] = new_res.get("initial_matrix")
        existing_res["local_optima"] = {**existing_res.get("local_optima", {}), **new_res.get("local_optima", {})}
        merged_trans = defaultdict(list, existing_res.get("transitions", {}))
        for src, dests in new_res.get("transitions", {}).items():
            merged_trans[src].extend(dests)
        existing_res["transitions"] = merged_trans

        new_data = existing_data

    # Atomic write
    tmp_path = full_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(new_data, f, default=custom_encoder, indent=2)
    os.replace(tmp_path, full_path)

    if is_final:
        logger.info(f"Saved final results to: {full_path}")
        cont_file = Path("Continuation") / f"{distribution}_{tsp_type}" / base_name
        if cont_file.exists():
            cont_file.unlink()
            logger.info(f"Removed continuation file {cont_file}")
    else:
        logger.info(f"Saved partial results to: {full_path}")

    #########################
    # Old code for reference
    #########################
#     base_name = f"city{citysize}_range{rang}_{mutation_strategy}.json"
#     folder = f"Results/{distribution}_{tsp_type}" if is_final else f"Continuation/{distribution}_{tsp_type}"
    
#     if not os.path.exists(folder):
#         os.makedirs(folder, exist_ok=True)
    
#     full_path = os.path.join(folder, base_name)

#     # Merge into existing file if it exists
#     if os.path.exists(full_path):
#         with open(full_path, "r") as f:
#             existing_data = json.load(f, object_hook=custom_decoder)
#         existing_data["time"] += time_spent

#         # Merge `all_iterations`
#         existing_iterations = existing_data["results"].get("all_iterations", [])
#         new_iterations = results.get("all_iterations", [])
#         existing_data["results"]["all_iterations"] = existing_iterations + new_iterations  # Append new values

#         # Update `last_matrix` and `hard_instances`
#         existing_data["results"]["last_matrix"] = results["last_matrix"]
#         existing_data["results"]["hard_instances"].update(results["hard_instances"])

#         # Update `initial_matrix` if it's not already there
#         if "initial_matrix" not in existing_data["results"]:
#             existing_data["results"]["initial_matrix"] = results["initial_matrix"]

#         # Update `local_optima` to be included in Continuation and Results
#         existing_local_optima = existing_data["results"].get("local_optima", {})
#         new_local_optima = results.get("local_optima", {})
#         existing_data["results"]["local_optima"] = {**existing_local_optima, **new_local_optima}
#         # Update `transitions` to be included in Continuation and Results
#         existing_trans = existing_data["results"].get("transitions", defaultdict(list))
#         existing_trans = defaultdict(list, existing_trans)
#         new_trans = results.get("transitions", defaultdict(list))
#         new_trans = defaultdict(list, new_trans)  # Ensure it's a defaultdict
#         for src, dests in new_trans.items():
#             existing_trans[src].extend(dests)
#         # convert back to regular dict before saving
#         existing_data["results"]["transitions"] = dict(existing_trans)
#     else:
#         existing_data = {
#             "time": time_spent,
#             "configuration": configuration,
#             "results": results
#         }
        

#     # Write
#     with open(full_path, "w") as f:
#         json.dump(existing_data, f, indent=2, default=custom_encoder)
    
#     #If final => remove from Continuation
#     if is_final:
#         logger.info(f"Saved final results to: {full_path}")
#         cont_path = os.path.join("Continuation", f"{distribution}_{tsp_type}", base_name)
#         if os.path.exists(cont_path):
#             os.remove(cont_path)
#             logger.info(f"Removed continuation file {cont_path}")
#     else:
#         logger.info(f"Saved partial results to: {full_path}")

# def load_full_results(cont_file):
#     """Loads entire results from a continuation file."""
#     with open(cont_file, "r") as f:
#         data = json.load(f, object_hook=custom_decoder)
#     return data.get("results", {})
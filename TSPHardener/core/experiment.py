import time
import os
from collections import defaultdict
import logging

from .helpers import initialize_matrix_and_hardest, run_litals_algorithm
from .mutate_tsp import apply_mutation
from utils.json_utils import save_partial, load_full_results
from utils.file_utils import get_result_path

logger = logging.getLogger(__name__)

def track_basin_transition(source_matrix, mutated_matrix, partial_results):
    source_hash = hash(source_matrix.tobytes())
    mutated_hash = hash(mutated_matrix.tobytes())
    partial_results["transitions"][source_hash].append(mutated_hash)

def run_single_experiment(configuration, citysize, rang, mutations):
    """
    Handles one (citysize, range) combination.
    If a results file is in Results, we skip.
    Otherwise we check Continuation for partial progress.

    partial_results is what is being passed around and saved.
    """
    start_time = time.time()
    partial_results = {
        "initial_matrix": [],
        "hard_instances": {},
        "local_optima": {},
        "transitions": defaultdict(list),
        "last_matrix": [],
        "all_iterations": []
    }

    # Load from partial or generate new
    hardest, matrix = initialize_matrix_and_hardest(citysize, rang, configuration)
    hardest_matrix = matrix.copy()

    # If no partial file existed => store the initial as iteration_0
    cont_file = get_result_path(
        citysize, 
        rang, 
        configuration["distribution"],
        configuration["generation_type"],
        configuration["mutation_type"],
        is_final=False
    )
    if os.path.exists(cont_file):
        try:
            existing_data = load_full_results(cont_file)
            partial_results["local_optima"] = existing_data.get("local_optima", {})
            partial_results["transitions"] = existing_data.get("transitions", defaultdict(list))
            partial_results["all_iterations"] = existing_data.get("all_iterations", [])
            partial_results["initial_matrix"] = existing_data.get("initial_matrix", [])
            partial_results["hard_instances"] = existing_data.get("hard_instances", {})
            partial_results["last_matrix"] = existing_data.get("last_matrix", [])
        except Exception as e:
            logger.error(f"Error loading continuation: {e}")
    else:
        # No partial => new initial TSP matrix => store it
        partial_results["initial_matrix"] = matrix.tolist() 

        # treat the initial matrix as a "hard" instance (iteration=0)
        partial_results["hard_instances"]["iteration_0"] = {
            "iterations": 0,
            "hardest": hardest,
            "optimal_tour": None,
            "optimal_cost": None,
            "matrix": matrix.tolist(),
            "is_hardest": True
        }
        partial_results["last_matrix"] = matrix.tolist()
        save_partial(
            configuration, 
            partial_results,
            citysize, 
            rang,
            time_spent=0,
            distribution=configuration["distribution"],
            tsp_type=configuration["generation_type"],
            mutation_strategy=configuration["mutation_type"],
            is_final=False
        )

    start_iter = len(partial_results["all_iterations"]) # start iteration from last generation
    for j in range(start_iter, mutations):
        iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
        if error:
            logger.error(f"Error in iteration {j}: {error}")
            continue
        else: # store iteration
            partial_results["all_iterations"].append(iterations)

        # Compare vs. hardest
        is_hardest = False
        if iterations > hardest:
            hardest = iterations
            hardest_matrix = matrix.copy()
            is_hardest = True

        # track local optima
        matrix_hash = hash(matrix.tobytes())
        partial_results["local_optima"][matrix_hash] = {
            "iterations": iterations,
            # "matrix": matrix.tolist(), # expensive so I am just using hash
            "cost": optimal_cost,
            "is_hardest": is_hardest
        }
        # log basin transitions
        track_basin_transition(hardest_matrix, matrix, partial_results)

        # mutate hardest_matrix for next iteration
        matrix = apply_mutation(hardest_matrix, configuration["mutation_type"],
                                configuration["generation_type"], rang,
                                configuration["distribution"])
        
        # Always store the last_matrix for continuation
        partial_results["last_matrix"] = matrix.tolist()
        
        # If it's a newly hardest, store it
        if is_hardest:
            iteration_key = f"iteration_{j+1}"
            partial_results["hard_instances"][iteration_key] = {
                "iterations": iterations,
                "hardest": hardest,
                "optimal_tour": optimal_tour,
                "optimal_cost": optimal_cost,
                "matrix": hardest_matrix.tolist(),
                "is_hardest": True
            }
        
        # Periodically (or when new hardest) do a partial save
        # j % 100 == 0
        # or "is_hardest" scenario:
        if (j % 100 == 0) or is_hardest:
            elapsed = time.time() - start_time
            save_partial(
                configuration, 
                partial_results, 
                citysize, 
                rang,
                time_spent=0,
                distribution=configuration["distribution"],
                tsp_type=configuration["generation_type"],
                mutation_strategy=configuration["mutation_type"],
                is_final=False
            )

    # Final save (whatever is left in partial_results)
    if partial_results["hard_instances"] or partial_results["last_matrix"]:
        elapsed = time.time() - start_time
        save_partial(
            configuration, 
            partial_results, 
            citysize, 
            rang,
            time_spent=elapsed,
            distribution=configuration["distribution"],
            tsp_type=configuration["generation_type"],
            mutation_strategy=configuration["mutation_type"],
            is_final=True
        )

    logger.info(f"Completed up to {mutations} mutations for citysize={citysize}, range={rang}.")

def experiment(_cities, _ranges, _mutations, continuations, distribution, tsp_type, mutation_strategy):    
    """
    Orchestrates an experiment with configuration
    """ 
    t0 = time.time()
    config = {
        "mutation_type": mutation_strategy,
        "generation_type": tsp_type,
        "distribution": distribution
    }
    for citysize in _cities:
        for rang in _ranges:
            if f"{citysize},{rang}" in continuations:
                run_single_experiment(config, citysize, rang, _mutations)
                continue

            results_file = get_result_path(
                citysize, 
                rang, 
                distribution, 
                tsp_type, 
                mutation_strategy, 
                is_final=True
            )
            if os.path.exists(results_file):
                logger.debug(f"Skipping citysize={citysize}, range={rang}, already in Results.")
                continue
            conf_with_params = {
                **config,
                "city_size": citysize,
                "range": rang
            }
            run_single_experiment(conf_with_params, citysize, rang, _mutations)
            
    logger.info(f"Total experiment duration: {time.time() - t0:.2f}s")

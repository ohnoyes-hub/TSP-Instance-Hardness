import time
from icecream import ic
import os

from .helpers import initialize_matrix_and_hardest, run_litals_algorithm
from .mutate_tsp import apply_mutation

from utils.json_utils import save_partial
from utils.file_utils import get_result_path


def run_single_experiment(configuration, citysize, rang, mutations):
    """
    Handles one (citysize, range) combination.
    If a results file is in Results, we skip.
    Otherwise we check Continuation for partial progress.

    The resulting structure is LIFO, i.e.
      partial_results = {
        "hard_instances": { ... },
        "last_matrix": [ ... ]
      }
    """
    start_time = time.time()
    partial_results = {
        "hard_instances": {},
        "last_matrix": []
    }

    # Load from partial or generate new
    hardest, matrix = initialize_matrix_and_hardest(citysize, rang, configuration)
    hardest_matrix = matrix.copy()

    # If no partial file existed => store the initial as iteration_0
    cont_file = os.path.join("Continuation",
                             f"{configuration['distribution']}_{configuration['generation_type']}",
                             f"city{citysize}_range{rang}_{configuration['mutation_type']}.json"
                            )
    if not os.path.exists(cont_file):
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
            configuration, partial_results, citysize, rang,
            time_spent=0,
            distribution=configuration["distribution"],
            tsp_type=configuration["generation_type"],
            mutation_strategy=configuration["mutation_type"],
            is_final=False
        )
        partial_results["hard_instances"] = {}  # reset in-memory dict

    non_improved_iterations = 0
    for j in range(mutations):
        prev_hardest = hardest
        iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
        if error:
            ic(f"Error in iteration {j}: {error}")
            continue

        # Compare vs. hardest
        is_hardest = False
        if iterations > hardest:
            hardest = iterations
            hardest_matrix = matrix.copy()
            is_hardest = True
            non_improved_iterations = 0
        else:
            non_improved_iterations += 1

        # Early stop if 10k consecutive non-improving
        if non_improved_iterations >= 10000:
            ic(f"Stopping early after {j} consecutive non-improving mutations.")
            break

        # Now mutate from the hardest matrix
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
                configuration, partial_results, citysize, rang,
                time_spent=0,
                distribution=configuration["distribution"],
                tsp_type=configuration["generation_type"],
                mutation_strategy=configuration["mutation_type"],
                is_final=False
            )
            # clear out just the "hard_instances" to keep memory down in Python
            partial_results["hard_instances"] = {}

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

    ic(f"Completed up to {mutations} mutations for citysize={citysize}, range={rang}.")

def experiment(_cities, _ranges, _mutations, continuations, distribution, tsp_type, mutation_strategy):    
    """
    Orchestrate experiments
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

            results_file = os.path.join("Results", f"{distribution}_{tsp_type}",
                                        f"city{citysize}_range{rang}_{mutation_strategy}.json")
            if os.path.exists(results_file):
                ic(f"Skipping citysize={citysize}, range={rang}, already in Results.")
                continue
            conf_with_params = {
                **config,
                "city_size": citysize,
                "range": rang
            }
            run_single_experiment(conf_with_params, citysize, rang, _mutations)
    ic(f"Total experiment duration: {time.time() - t0:.2f}s")

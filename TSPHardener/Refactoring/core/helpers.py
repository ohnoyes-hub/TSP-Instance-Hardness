from pathlib import Path
from utils.json_utils import save_partial, load_partial
from collections import defaultdict
from core.generate_tsp import TSPBuilder
from core.algorithm import get_minimal_route


def get_continuation_path(config):
    """
    Returns the path to the continuation file based on the configuration.
    """
    folder = Path("Continuation") / f"{config.distribution}_{config.generation_type}"
    folder.mkdir(parents=True, exist_ok=True)
    base_name = f"city{config.city_size}_range{config.control}_{config.mutation_type}.json"
    return folder / base_name

def load_or_initialize(config):
    """
    Loads partial experiment state or initializes a fresh TSP instance.

    """
    path = get_continuation_path(config)
    if path.exists():
        hardest, matrix, local_optima, transitions = load_partial(str(path))
        return hardest, matrix, local_optima, transitions
    inst = (
        TSPBuilder()
        .set_city_size(config.city_size)
        .set_generation_type(config.generation_type)
        .set_distribution(config.distribution)
        .set_control(config.control)
        .build()
    )
    return 0, inst.matrix, {}, defaultdict(list)

def save_results(config, results: dict, elapsed_time: float, is_final: bool = False) -> None:
    """
    Saves results (partial or final) to disk, merging with existing data if needed.
    """
    # Use json_utils.save_partial under the hood
    save_partial(
        configuration=config.__dict__,
        results=results,
        citysize=config.city_size,
        rang=config.control,
        time_spent=elapsed_time,
        distribution=config.distribution,
        tsp_type=config.generation_type,
        mutation_strategy=config.mutation_type,
        is_final=is_final
    )
    print(f"File saved to {get_continuation_path(config)}")

###########
# OLD CODE
###########

import numpy as np
# import time
import os
import logging
# from .generate_tsp import TSPBuilder, TSPInstance
# from .mutate_tsp import get_mutation_strategy
# from utils.json_utils import save_partial
# from utils.json_utils import load_partial
from config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

def run_litals_algorithm(matrix):
    """Run Lital's algorithm and return results with error handling."""
    try:
        iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
        return iterations, optimal_tour, optimal_cost, None
    except Exception as e:
        return None, None, None, e

def initialize_state_or_load(config: ExperimentConfig) -> tuple[int, np.ndarray]:
    """Load saved state or initialize new TSP instance"""
    cont_file = _get_continuation_filename(config)
    if os.path.exists(cont_file):
        try:
            hardest, matrix = load_partial(cont_file)
            return hardest, matrix
        except Exception as e:
            logger.error(f"Error loading partial: {e}")
    
    # Fallback to new instance
    instance = _create_tsp_instance(config)
    return 0, instance.matrix

def _get_continuation_filename(config: ExperimentConfig) -> str:
    """Generate standardized continuation filename."""
    return os.path.join(
        "Continuation",
        f"{config.distribution}_{config.generation_type}",
        f"city{config.citysize}_range{config.rang}_{config.mutation_type}.json"
    )

def _create_tsp_instance(config: ExperimentConfig):
    """Centralized TSP instance creation."""
    return (
        TSPBuilder()
        .set_city_size(config.citysize)
        .set_generation_type(config.generation_type)
        .set_distribution(config.distribution)
        .set_control(config.rang)
        .build()
    )

def initialize_matrix_or_hardest(citysize, rang, config):
    """Check if we have partial data; else generate new TSP."""
    cont_file = os.path.join("Continuation",
                             f"{config['distribution']}_{config['generation_type']}",
                             f"city{citysize}_range{rang}_{config['mutation_type']}.json")
    if os.path.exists(cont_file):
        try:
            hardest, matrix = load_partial(cont_file)
            return hardest, matrix
        except Exception as e:
            logger.error(f"Error loading partial: {e}")
    
    # fallback => new matrix
    builder = (
        TSPBuilder()
        .set_city_size(citysize)
        .set_generation_type(config['generation_type'])
        .set_distribution(config['distribution'])
        .set_control(rang)
    ) 
    instance = builder.build()
    return 0, instance.matrix

# def process_mutation_iteration(j, matrix, hardest, hardest_matrix, 
#                                mutation_type, generation_type, rang, distribution):
#     """Process a single mutation iteration and return updated state."""
#     iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
#     if error:
#         logger.error(f"Error in iteration {j}: {error}")
#         return hardest, hardest_matrix, matrix, None  # No results to record
    
#     iteration_result = {
#         "iterations": iterations,
#         "hardest": hardest,
#         "optimal_tour": optimal_tour,
#         "optimal_cost": optimal_cost,
#         "matrix": matrix.tolist(),
#         "is_hardest": False
#     }
    
#     # Update hardest matrix if needed
#     if iterations > hardest:
#         hardest = iterations
#         hardest_matrix = matrix.copy()
#         iteration_result["is_hardest"] = True
    
#     # Apply mutation to the hardest matrix
#     # TODO: check if .copy() is needed(copy is n)
#     tsp_instance = TSPInstance(hardest_matrix, generation_type)
#     #tsp_instance = TSPInstance(hardest_matrix.copy(), generation_type)
#     strategy = get_mutation_strategy(mutation_type, generation_type, distribution, rang)
#     new_matrix = strategy.mutate(tsp_instance).matrix
#     return hardest, hardest_matrix, new_matrix, iteration_result
    
# def handle_saving(configuration, results, citysize, rang, 
#                   start_time, should_save, is_final=False):
#     """Save results to coninuation or final file."""
#     if should_save:
#         elapsed_time = time.time() - start_time
#         save_partial(
#             configuration, 
#             results, 
#             citysize, 
#             rang,
#             elapsed_time,
#             configuration["distribution"],
#             configuration["generation_type"],
#             configuration["mutation_type"],
#             is_final=is_final
#         )
#         return {}
#     return results

# def save_results(
#     config: ExperimentConfig,
#     results: list[dict],
#     elapsed_time: float,
#     is_final: bool = False
# ) -> None:
#     """Centralized result saving."""
#     save_partial(
#         config=config.__dict__,
#         results=results,
#         citysize=config.citysize,
#         elapsed_time=elapsed_time,
#         is_final=is_final
#     )
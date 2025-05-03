import numpy as np
import os
import time
import logging
from .generate_tsp import TSPBuilder, TSPInstance
from .mutate_tsp import get_mutation_strategy
from .algorithm import get_minimal_route
from utils.json_utils import save_partial
from utils.json_utils import load_partial

logger = logging.getLogger(__name__)

def initialize_matrix_and_hardest(citysize, rang, config):
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
    builder = TSPBuilder().set_city_size(citysize).set_generation_type(config['generation_type']).set_distribution(config['distribution']).set_control(rang)
    tsp_instance = builder.build()
    hardest = 0
    return hardest, tsp_instance.matrix

def process_mutation_iteration(j, matrix, hardest, hardest_matrix, 
                               mutation_type, generation_type, rang, distribution):
    """Process a single mutation iteration and return updated state."""
    iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
    if error:
        logger.error(f"Error in iteration {j}: {error}")
        return hardest, hardest_matrix, matrix, None  # No results to record
    
    iteration_result = {
        "iterations": iterations,
        "hardest": hardest,
        "optimal_tour": optimal_tour,
        "optimal_cost": optimal_cost,
        "matrix": matrix.tolist(),
        "is_hardest": False
    }
    
    # Update hardest matrix if needed
    if iterations > hardest:
        hardest = iterations
        hardest_matrix = matrix.copy()
        iteration_result["is_hardest"] = True
    
    # Apply mutation to the hardest matrix
    tsp_instance = TSPInstance(hardest_matrix.copy(), generation_type)
    strategy = get_mutation_strategy(mutation_type, generation_type, distribution, rang)
    new_matrix = strategy.mutate(tsp_instance).matrix
    return hardest, hardest_matrix, new_matrix, iteration_result

def run_litals_algorithm(matrix):
    """Run Lital's algorithm and return results with error handling."""
    try:
        iterations, optimal_tour, optimal_cost = get_minimal_route(matrix)
        return iterations, optimal_tour, optimal_cost, None
    except Exception as e:
        return None, None, None, e
    
def handle_saving(configuration, results, citysize, rang, start_time, continuations, should_save, is_final=False):
    if should_save:
        elapsed_time = time.time() - start_time
        save_partial(
            configuration, results, citysize, rang,
            elapsed_time,
            is_final=is_final
        )
        return {}
    return results
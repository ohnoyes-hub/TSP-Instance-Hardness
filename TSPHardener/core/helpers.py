import numpy as np
import os
import time
import logging
from .generate_tsp import TSPBuilder
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
    tsp_instance = (
            TSPBuilder()
            .set_city_size(citysize)
            .set_generation_type(config["generation_type"])
            .set_distribution(config["distribution"])
            .set_control(rang)
            .build()
        )

    matrix = tsp_instance.matrix
    hardest = 0
    return hardest, matrix

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
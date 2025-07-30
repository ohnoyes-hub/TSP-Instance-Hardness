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

def run_single_phase_transition_experiment(configuration, citysize, rang, mutations):
    """
    Modified Hill-climber above to not Hill-climb where each iteration will new instance instead of mutating the hardest.
    Replicate phase transition experiment from prior literature not using hill-climbing.
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
            partial_results["hard_instances"] = existing_data.get("hard_instances", {})
            partial_results["all_iterations"] = existing_data.get("all_iterations", [])
            # Ignore transitions, initial_matrix, last_matrix as they are not used
        except Exception as e:
            logger.error(f"Error loading continuation: {e}")
    else:
        # No initial setup needed for new matrices
        pass

    start_iter = len(partial_results["all_iterations"])
    hardest = 0
    hardest_matrix = None

    for j in range(start_iter, mutations):
        # Generate a new matrix for each iteration
        _, matrix = initialize_matrix_and_hardest(citysize, rang, configuration)
        
        # Run the algorithm on the new matrix
        iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
        if error:
            logger.error(f"Error in iteration {j}: {error}")
            continue
        else:
            partial_results["all_iterations"].append(iterations)

        # Check if this is the hardest instance so far
        is_hardest = False
        if iterations > hardest:
            hardest = iterations
            hardest_matrix = matrix.copy()
            is_hardest = True

        # Track local optima (no transitions)
        matrix_hash = hash(matrix.tobytes())
        partial_results["local_optima"][matrix_hash] = {
            "iterations": iterations,
            "cost": optimal_cost,
            "is_hardest": is_hardest
        }

        # Store as a hard instance if it's the hardest
        if is_hardest:
            iteration_key = f"iteration_{j}"
            partial_results["hard_instances"][iteration_key] = {
                "iterations": iterations,
                "hardest": hardest,
                "optimal_tour": optimal_tour,
                "optimal_cost": optimal_cost,
                "matrix": hardest_matrix.tolist(),
                "is_hardest": True
            }

        # Periodically save partial results
        if (j % 100 == 0) or is_hardest:
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
                is_final=False
            )

    # Final save
    if partial_results["hard_instances"]:
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

    logger.info(f"Completed up to {mutations} new instances for citysize={citysize}, range={rang}.")

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
            run_single_phase_transition_experiment(conf_with_params, citysize, rang, _mutations)
            
    logger.info(f"Total experiment duration: {time.time() - t0:.2f}s")


# import time
# import os
# from collections import defaultdict
# import logging
# from abc import ABC, abstractmethod
# from core.helpers import load_or_initialize, run_litals_algorithm, save_results
# from core.generate_tsp import TSPBuilder, TSPInstance
# from core.mutate_tsp import get_mutation_strategy
# from utils.file_utils import get_result_path
# from dataclasses import dataclass
# from typing import List, Union
# import numpy as np

# logger = logging.getLogger(__name__)
    
# # def track_basin_transition(source_matrix, mutated_matrix, partial_results):
# #     source_hash = hash(source_matrix.tobytes())
# #     mutated_hash = hash(mutated_matrix.tobytes())
# #     partial_results["transitions"][source_hash].append(mutated_hash)

# @dataclass
# class ExperimentConfig:
#     """Defines runtime experiement executution""" 
#     city_size: int
#     generation_type: str
#     distribution: str
#     mutation_type: str
#     control: List[Union[float, int]]

# class ExperimentProtoype(ABC):
#     @abstractmethod
#     def clone(self) -> 'ExperimentProtoype':
#         """Clone the experiment instance."""

# @dataclass
# class ExperimentState(ExperimentProtoype):
#     """Defines the state of the experiment"""
#     current_matrix: np.ndarray
#     hardest_matrix: np.ndarray
#     mutation_type: str
#     control: List[Union[float, int]]
#     hardest_count: int

#     def clone(self) -> 'ExperimentState':
#         """Clone the experiment state."""
#         return ExperimentState(
#             current_matrix=self.current_matrix.copy(),
#             hardest_matrix=self.hardest_matrix.copy(),
#             mutation_type=self.mutation_type,
#             control=self.control,
#             hardest_count=self.hardest_count
#         )

# class ExperimentProcessor:
#     """Handles the mutation and processing of the experiment."""
#     def __init__(self, config: ExperimentConfig):
#         self.config = config
#         self.strategy = get_mutation_strategy(
#             config.mutation_type,
#             config.generation_type,
#             config.distribution,
#             config.control
#         )
    
#     def process_iteration(self, state: ExperimentState) -> tuple[ExperimentState, dict]:
#         """Run one iteration of algorithm and mutation."""
#         iterations, tour, cost, error = run_litals_algorithm(state.current_matrix)
#         if error:
#             logger.error(f"Iteration failed: {error}")
#             return state, {'error': error}
        
#         is_hardest = iterations > state.hardest_count
#         if is_hardest:
#             state.hardest_count = iterations
#             state.hardest_matrix = state.current_matrix.copy()
        
#         result = {
#             'iterations': iterations,
#             'hardest_count': state.hardest_count,
#             'optimal_tour': tour,
#             'optimal_cost': cost,
#             'is_hardest': is_hardest
#         }
        
#         # mutate for next iteration
#         tsp_inst = TSPInstance(state.current_matrix, self.config.generation_type)
#         mutated = self.strategy.mutate(tsp_inst).matrix
#         state.current_matrix = mutated
        
#         return state, result
    
#     def _mutate_hardest(self, matrix: np.ndarray) -> np.ndarray:
#         """Apply mutation to the hardest matrix."""
#         return self.strategy.mutate(TSPInstance(matrix, self.config.generation_type)).matrix
    
#     def _update_state(self, iterations: int, state: ExperimentState) -> ExperimentState:
#         """Update hardest matrix if needed."""
#         if iterations > state.hardest_count:
#             return ExperimentState(
#                 current_matrix=state.current_matrix,
#                 hardest_matrix=state.current_matrix,
#                 hardest_count=iterations
#             )
#         return state
    
#     def _create_result_dict(self, iterations: int, optimal_tour, optimal_cost, state: ExperimentState) -> dict:
#         return {
#             "iterations": iterations,
#             "hardest": state.hardest_count,
#             "optimal_tour": optimal_tour,
#             "optimal_cost": optimal_cost,
#             "is_hardest": iterations > state.hardest_count
#         }


# class Experiment:
#     def __init__(self, config: ExperimentConfig):
#         self.config = config
#         hardest, matrix, local_optima, transitions = load_or_initialize(config)
#         self.state = ExperimentState(
#             mutation_type=config.mutation_type,
#             control=config.control,
#             current_matrix=matrix,
#             hardest_matrix=matrix,
#             hardest_count=hardest
#         )
#         self.processor = ExperimentProcessor(config)
    
#     def run(self, iterations: int) -> list[dict]:
#         results = []
#         for i in range(iterations):
#             self.state, record = self.processor.process_iteration(self.state)
#             results.append(record)
#         return results

# def experiment(
#     city_sizes: List[int],
#     ranges: List[int],
#     num_mutations: int,
#     continuations: List[str],
#     distribution: str,
#     tsp_type: str,
#     mutation_strategy: str
# ) -> None:
#     config = ExperimentConfig(city_size=30, generation_type='euclidean', distribution='uniform', mutation_type='swap', control=10)
#     exp = Experiment(config)
#     start_time = time.time()
#     results = exp.run(num_mutations)
#     elapsed_time = time.time() - start_time
#     save_results(config, results, elapsed_time, is_final=True)

# experiment(
#     city_sizes=[8],
#     ranges=[10],
#     num_mutations=10,
#     continuations=[],
#     distribution='uniform',
#     tsp_type='asymmetric',
#     mutation_strategy='swap'
#     )

# ###############
# # OLD CODE - working prior to refactoring to tsp builder and mutation strategy.
# ###############

# # def run_single_experiment(configuration, citysize, rang, mutations):
# #     """
# #     Handles one (citysize, range) combination.
# #     If a results file is in Results, we skip.
# #     Otherwise we check Continuation for partial progress.

# #     partial_results is what is being passed around and saved.
# #     """
# #     start_time = time.time()
# #     partial_results = {
# #         "initial_matrix": [],
# #         "hard_instances": {},
# #         "local_optima": {},
# #         "transitions": defaultdict(list),
# #         "last_matrix": [],
# #         "all_iterations": []
# #     }

# #     # Load from partial or generate new
# #     hardest, matrix = initialize_matrix_or_hardest(citysize, rang, configuration)
# #     hardest_matrix = matrix.copy()

# #     # mutation strategy
# #     mutation_strategy = get_mutation_strategy(
# #         configuration["mutation_type"],
# #         configuration["generation_type"],
# #         configuration["distribution"],
# #         rang
# #     )

# #     # If no partial file existed => store the initial as iteration_0
# #     cont_file = get_result_path(
# #         citysize, 
# #         rang, 
# #         configuration["distribution"],
# #         configuration["generation_type"],
# #         configuration["mutation_type"],
# #         is_final=False
# #     )
# #     if os.path.exists(cont_file):
# #         try:
# #             existing_data = load_full_results(cont_file)
# #             # partial_results["local_optima"] = existing_data.get("local_optima", {})
# #             # partial_results["transitions"] = existing_data.get("transitions", defaultdict(list))
# #             partial_results["all_iterations"] = existing_data.get("all_iterations", [])
# #             partial_results["initial_matrix"] = existing_data.get("initial_matrix", [])
# #             partial_results["hard_instances"] = existing_data.get("hard_instances", {})
# #             partial_results["last_matrix"] = existing_data.get("last_matrix", [])
# #         except Exception as e:
# #             logger.error(f"Error loading continuation: {e}")
# #     else:
# #         # No partial => new initial TSP matrix => store it
# #         partial_results["initial_matrix"] = matrix.tolist() 

# #         # treat the initial matrix as a "hard" instance (iteration=0)
# #         partial_results["hard_instances"]["iteration_0"] = {
# #             "iterations": 0,
# #             "hardest": hardest,
# #             "optimal_tour": None,
# #             "optimal_cost": None,
# #             "matrix": matrix.tolist(),
# #             "is_hardest": True
# #         }
# #         partial_results["last_matrix"] = matrix.tolist()
# #         save_partial(
# #             configuration, 
# #             partial_results,
# #             citysize, 
# #             rang,
# #             time_spent=0,
# #             distribution=configuration["distribution"],
# #             tsp_type=configuration["generation_type"],
# #             mutation_strategy=configuration["mutation_type"],
# #             is_final=False
# #         )

# #     start_iter = len(partial_results["all_iterations"]) # start iteration from last generation
# #     for j in range(start_iter, mutations):
# #         iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
# #         if error:
# #             logger.error(f"Error in iteration {j}: {error}")
# #             continue
# #         else: # store iteration
# #             partial_results["all_iterations"].append(iterations)

# #         # Compare vs. hardest
        

# #         # track local optima
# #         # matrix_hash = hash(matrix.tobytes())
# #         # partial_results["local_optima"][matrix_hash] = {
# #         #     "iterations": iterations,
# #         #     # "matrix": matrix.tolist(), # expensive so I am just using hash
# #         #     "cost": optimal_cost,
# #         #     "is_hardest": is_hardest
# #         # }
# #         # # log basin transitions
# #         # track_basin_transition(hardest_matrix, matrix, partial_results)

# #         # Mutate the hardest matrix for hill-climbing(expect for random_sampling which creates new instance)
# #         tsp_instance = TSPInstance(hardest_matrix.copy(), configuration["generation_type"])
# #         mutation_strategy.mutate(tsp_instance)
# #         new_matrix = tsp_instance.matrix 

# #         # Always store the last_matrix for continuation
# #         partial_results["last_matrix"] = matrix.tolist()
# #         is_hardest = False

        
# #         if iterations > hardest:
# #             hardest = iterations
# #             is_hardest = True
        
# #         matrix = new_matrix
# #         # # If it's a newly hardest, store it
# #         # if is_hardest:
# #         #     iteration_key = f"iteration_{j+1}"
# #         #     partial_results["hard_instances"][iteration_key] = {
# #         #         "iterations": iterations,
# #         #         "hardest": hardest,
# #         #         "optimal_tour": optimal_tour,
# #         #         "optimal_cost": optimal_cost,
# #         #         "matrix": hardest_matrix.tolist(),
# #         #         "is_hardest": True
# #         #     }
        
# #         # Periodically (or when new hardest) do a partial save
# #         # j % 100 == 0
# #         # or "is_hardest" scenario:
# #         if (j % 100 == 0) or is_hardest:
# #             elapsed = time.time() - start_time
# #             save_partial(
# #                 configuration, 
# #                 partial_results, 
# #                 citysize, 
# #                 rang,
# #                 time_spent=0,
# #                 distribution=configuration["distribution"],
# #                 tsp_type=configuration["generation_type"],
# #                 mutation_strategy=configuration["mutation_type"],
# #                 is_final=False
# #             )

# #     # Final save (whatever is left in partial_results)
# #     if partial_results["hard_instances"] or partial_results["last_matrix"]:
# #         elapsed = time.time() - start_time
# #         save_partial(
# #             configuration, 
# #             partial_results, 
# #             citysize, 
# #             rang,
# #             time_spent=elapsed,
# #             distribution=configuration["distribution"],
# #             tsp_type=configuration["generation_type"],
# #             mutation_strategy=configuration["mutation_type"],
# #             is_final=True
# #         )

# #     logger.info(f"Completed up to {mutations} mutations for citysize={citysize}, range={rang}.")

# # def run_single_phase_transition_experiment(configuration, citysize, rang, mutations):
# #     """
# #     Modified Hill-climber above to not Hill-climb where each iteration will new instance instead of mutating the hardest.
# #     Replicate phase transition experiment from prior literature not using hill-climbing.
# #     """
# #     start_time = time.time()
# #     partial_results = {
# #         "initial_matrix": [],
# #         "hard_instances": {},
# #         "local_optima": {},
# #         "transitions": defaultdict(list),
# #         "last_matrix": [],
# #         "all_iterations": []
# #     }

# #     # Load from partial or generate new
# #     cont_file = get_result_path(
# #         citysize, 
# #         rang, 
# #         configuration["distribution"],
# #         configuration["generation_type"],
# #         configuration["mutation_type"],
# #         is_final=False
# #     )
# #     if os.path.exists(cont_file):
# #         try:
# #             existing_data = load_full_results(cont_file)
# #             partial_results["local_optima"] = existing_data.get("local_optima", {})
# #             partial_results["hard_instances"] = existing_data.get("hard_instances", {})
# #             partial_results["all_iterations"] = existing_data.get("all_iterations", [])
# #             # Ignore transitions, initial_matrix, last_matrix as they are not used
# #         except Exception as e:
# #             logger.error(f"Error loading continuation: {e}")
# #     else:
# #         # No initial setup needed for new matrices
# #         pass

# #     start_iter = len(partial_results["all_iterations"])
# #     hardest = 0
# #     hardest_matrix = None

# #     for j in range(start_iter, mutations):
# #         # Generate a new matrix for each iteration
# #         _, matrix = initialize_matrix_or_hardest(citysize, rang, configuration)
        
# #         # Run the algorithm on the new matrix
# #         iterations, optimal_tour, optimal_cost, error = run_litals_algorithm(matrix)
# #         if error:
# #             logger.error(f"Error in iteration {j}: {error}")
# #             continue
# #         else:
# #             partial_results["all_iterations"].append(iterations)

# #         # Check if this is the hardest instance so far
# #         is_hardest = False
# #         if iterations > hardest:
# #             hardest = iterations
# #             hardest_matrix = matrix.copy()
# #             is_hardest = True

# #         # Track local optima (no transitions)
# #         matrix_hash = hash(matrix.tobytes())
# #         partial_results["local_optima"][matrix_hash] = {
# #             "iterations": iterations,
# #             "cost": optimal_cost,
# #             "is_hardest": is_hardest
# #         }

# #         # Store as a hard instance if it's the hardest
# #         if is_hardest:
# #             iteration_key = f"iteration_{j}"
# #             partial_results["hard_instances"][iteration_key] = {
# #                 "iterations": iterations,
# #                 "hardest": hardest,
# #                 "optimal_tour": optimal_tour,
# #                 "optimal_cost": optimal_cost,
# #                 "matrix": hardest_matrix.tolist(),
# #                 "is_hardest": True
# #             }

# #         # Periodically save partial results
# #         if (j % 100 == 0) or is_hardest:
# #             elapsed = time.time() - start_time
# #             save_partial(
# #                 configuration, 
# #                 partial_results,
# #                 citysize, 
# #                 rang,
# #                 time_spent=elapsed,
# #                 distribution=configuration["distribution"],
# #                 tsp_type=configuration["generation_type"],
# #                 mutation_strategy=configuration["mutation_type"],
# #                 is_final=False
# #             )

# #     # Final save
# #     if partial_results["hard_instances"]:
# #         elapsed = time.time() - start_time
# #         save_partial(
# #             configuration, 
# #             partial_results,
# #             citysize, 
# #             rang,
# #             time_spent=elapsed,
# #             distribution=configuration["distribution"],
# #             tsp_type=configuration["generation_type"],
# #             mutation_strategy=configuration["mutation_type"],
# #             is_final=True
# #         )

# #     logger.info(f"Completed up to {mutations} new instances for citysize={citysize}, range={rang}.")

# # def experiment(city_sizes, ranges, num_mutations,continuations, distribution, tsp_type, mutation_strategy):    
# #     """
# #     Orchestrates an experiment with configuration
# #     """ 
# #     t0 = time.time()
# #     config = {
# #         "mutation_type": mutation_strategy,
# #         "generation_type": tsp_type,
# #         "distribution": distribution
# #     }
# #     for city_size in city_sizes:
# #         for rang in ranges:
# #             key = f"{city_size},{rang}"

# #             final_path = get_result_path(
# #                 city_size, rang,
# #                 distribution, tsp_type, mutation_strategy,
# #                 is_final=True
# #             )
# #             if os.path.exists(final_path):
# #                 logger.debug(
# #                     f"Skipping city={city_size}, range={rang}; final results already exist."
# #                 )
# #                 continue
        
# #             if key in continuations:
# #                 logger.info(f"Resuming hill-climb for city={city_size}, range={rang}.")
# #                 run_single_experiment(config, city_size, rang, num_mutations)
# #             else:
# #                 logger.info(f"Starting phase-transition experiment for city={city_size}, range={rang}.")
# #                 run_single_phase_transition_experiment(config, city_size, rang, num_mutations)
            
# #     logger.info(f"Total experiment duration: {time.time() - t0:.2f}s")

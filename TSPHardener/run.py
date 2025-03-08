import csv
import subprocess
from multiprocessing import Pool
import json
from formulation.validate import ExperimentConfig, load_configs
import logging
from utils.log_util import init_logger

def run_experiment(config):
    # params
    city_size = config.size
    tsp_type = config.tsp_type
    distribution = config.distribution
    mutation_strategy = config.mutation_strategy
    ranges = config.ranges
    mutations = config.mutations
    continuation = config.continuation
    
    # Formulation build command
    cmd = [
        'python3', '-m', 'main',
        json.dumps([city_size]),
        json.dumps(ranges),
        str(mutations),
        continuation,
        '--tsp_type', tsp_type,
        '--distribution', distribution,
        '--mutation_strategy', mutation_strategy
    ]
    
    logging.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        logging.info(f"Completed: {cmd}")
        logging.debug(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in {cmd}:", f"{e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error in {' '.join(cmd)}: {str(e)}")

if __name__ == '__main__':
    init_logger('run.log')
    configs = load_configs('tsp-formulations.csv')
    
    with Pool(processes=12) as pool:
        pool.map(run_experiment, configs)
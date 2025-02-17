import csv
import subprocess
from multiprocessing import Pool
import json
from formulation.validate import ExperimentConfig, load_configs
from icecream import ic

def run_experiment(config):
    # params
    city_size = config.size
    tsp_type = config.tsp_type
    distribution = config.distribution
    mutation_strategy = config.mutation_strategy
    ranges = config.ranges
    mutations = config.mutations
    continuation = config.continuation
    
    # Build command
    cmd = [
        'python3', '-m', 'experiment',
        json.dumps([city_size]),
        json.dumps(ranges),
        str(mutations),
        continuation,
        '--tsp_type', tsp_type,
        '--distribution', distribution,
        '--mutation_strategy', mutation_strategy
    ]
    
    ic(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        ic(f"Completed: {cmd}", f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        ic(f"Error in {cmd}:", f"{e.stderr}")
    except Exception as e:
        ic(f"Unexpected error:", f"{str(e)}")

if __name__ == '__main__':
    configs = load_configs('tsp-formulations.csv')
    
    # Adjust the number of processes as needed
    with Pool(processes=4) as pool:
        pool.map(run_experiment, configs)
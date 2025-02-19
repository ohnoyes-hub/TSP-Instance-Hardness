import csv
import multiprocessing
from experiment import experiment

def load_configurations(file_path):
    """Load experiment configurations from CSV file."""
    configurations = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            configurations.append(row)
    return configurations

def validate_and_convert_config(config):
    """Convert CSV parameters to experiment-compatible values."""
    mutation_type_mapping = {
        "scramble": "scramble",
        "swap": "swap",
        "inplace": "wouter"
    }
    
    try:
        # Convert parameters
        city_size = int(config["City Size"])
        tsp_variant = config["TSP Variant"].lower()
        distribution = config["Cost Distribution"].lower()
        mutation_type = config["Mutation Type"].lower()
        
        # Handle mutation type mapping
        mutation_type = mutation_type_mapping.get(mutation_type, mutation_type)
        
        # Set range based on distribution
        range_val = [1000] if distribution == "uniform" else [1]  # Example values
        
        return {
            "sizes": [city_size],
            "ranges": range_val,
            "mutations": 500,  # Fixed or read from config
            "tsp_type": tsp_variant,
            "distribution": distribution,
            "mutation_strategy": mutation_type
        }
    except Exception as e:
        print(f"Invalid config {config}: {str(e)}")
        return None

def run_single_experiment(params):
    """Wrapper function for multiprocessing"""
    if params is None:
        return
    print(f"Starting experiment: {params}", flush=True)
    experiment(
        _cities=params["sizes"],
        _ranges=params["ranges"],
        _mutations=params["mutations"],
        _continuations=[],
        generation_type=params["tsp_type"],
        distribution=params["distribution"],
        mutation_type=params["mutation_strategy"]
    )

if __name__ == "__main__":
    # Load and validate configurations
    configs = load_configurations("tsp-formulations.csv")
    valid_params = [validate_and_convert_config(c) for c in configs]
    valid_params = [p for p in valid_params if p is not None]

    # Run experiments in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run_single_experiment, valid_params)

# make sure that there is a flag in the the result data save that a specific formualtion run is complete or incomplete(no improvements after 100000 generations). 
# If the formulation experiment is incomplete then I would like the data to load from its previous configuration.
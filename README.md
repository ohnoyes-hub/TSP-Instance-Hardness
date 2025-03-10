# Where Really Hard Traveling Salesman Problems Are

This repository contains source code for the experiments and analysis presented in Where Really Hard Traveling Salesman Problems Are, a thesis by Thomas Sijpkens. It expands upon (Wouter Knibbe's thesis)[https://github.com/WouterKnibbe/ATSP_hillForHard] by exploring a more diverse set of configurations and mutation strategies for generating hard TSP instances using hill climbing.
The study focuses on comparing how well different TSP configurations perform when used in hill climbing for hardness instance generation, with particular attention to:

- TSP types (e.g., Euclidean, etc.)
- Mutation strategies (e.g., scramble, etc.)
- Cost distributions (e.g., uniform, lognormal)

Local optima and transition paths are recorded during hill climbing, and these are used to build a local optima network. This network helps illustrate how effectively various hill-climbing algorithms can escape local optima across different configurations.

# Dependencies

To install all required libraries for both experiments and analysis:

`pip install -r requirements.txt`

# Structure`

.
├── TSPHardener
│   ├── main.py
│   ├── run.py
│   ├── core/
│   └── utils/
├── analysis/
│   └── (scripts for plotting and statistics)
├── tsp-formulation.csv
└── requirements.txt


TSPHardener is where `main.py` and `run.py`.  
- `main.py` : Runs a single experiment formulation with specified command-line arguments.
  Example usage
 `python3 -m main "[10]" "[5,10,15]" 100 "" --tsp_type "euclidean" --distribution "uniform" --mutation_strategy "scramble"`
  In this example:
    - City size: 10
    - Control parameter: [5,10,15]
    - 100 generations
    - TSP type: Euclidean
    - Cost distribution: uniform
    - Mutation strategy: scramble
  
- `run.py` : Reads all formulations from `tsp-formulation.csv`. `run.py` and executes them in parallel using a thread pool. Each formulation is passed to a hill-climber run.
- `core` :  Contains the core logic for the hill-climber.
- `utils` : Utility functions for experiment data handling.
- `analysis` :  Independent plotting and statistical scripts which load their data from `load_json`. These scripts provide visualizations and quantitative insights into the experiment.

# Where Really Hard Traveling Salesman Problems Are

This repository contains the source code for experiments and analyses presented in the thesis Where Really Hard Traveling Salesman Problems Are by Thomas Sijpkens. It extends and builds upon the prior work of Wouter Knibbe's thesis, introducing a broader range of TSP configurations and more varied mutation strategies for both hill climbing and random sampling methods.

To effectively leverage computational resources, each experimental configuration is designed to be embarrassingly parallel, resulting in a total of 620 independent experimental runs.

Continuing Work

During the hill climbing process, local optima and transition paths are systematically recorded to construct local optima networks. These networks provide detailed insights into the evolutionary trajectories and characteristics of TSP instances, clarifying how different instances evolve under various mutation strategies and random sampling approaches.

# Dependencies

To install all required libraries for both experiments and analysis:

`pip install -r requirements.txt`

# Structure`

├── TSPHardener

│ ├── main.py

│ ├── run.py

│ ├── core/

│ └── utils/

├── analysis/

│ ├── analysis

│ └── util

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


Random Sampling experiment:
100 phase transition experiments

lognormal (25) * tsp_type (2) * city_size(2)

80 phase transition experiments

uniform (20) * tsp_type (2) * city_size(2)

Hill-climber experiment:
Type of run for which experiment: fine grain detail(feedback)

lognormal 
25 * 2 * 2 * 3 = 300
uniform 
20 * 2 * 2 * 3 = 240

average cell value of hardest intances. - difference between local optima

TODO:
1. describe random sampling
2. random sampling figures
3. diffrence between initial and evolved instance of hill-climber (lital iterations)
4. tiqness, average, std of cell values of matrices. 
5. make github public

Experiments to redo:
hill-climber: size30, inplace, euclidean, lognormal
hill-climber: size20, scramble, euclidean, lognormal
hill-climber: size30, scramble, euclidean, lognormal
hill-climber: size30, swap, euclidean, lognormal

phase-transition: size20, euclidean, lognormal
phase-transition: size30, euclidean, lognormal
phase-transition: size30, euclidean, uniform

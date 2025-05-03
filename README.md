# Where Really Hard Traveling Salesman Problems Are

This repository contains the source code for experiments and analyses presented in the thesis Where Really Hard Traveling Salesman Problems Are by Thomas Sijpkens. It extends and builds upon the prior work of Wouter Knibbe's thesis, introducing a broader range of TSP configurations and more varied mutation strategies for both hill climbing and random sampling methods.

To effectively leverage computational resources, each experimental configuration is designed run in parallel, resulting in a total of 620 independent experimental runs.

---

## Running Experiments

### Full Experiment
To initiate a complete set of experiments based on `tsp-formulation.csv`, run:

```bash
python3 -m TSPHardener.run
```
Each row in the `tsp-formulation.csv` corresponds to a single experiment and utilizes one thread from a thread pool. To adjust the number of threads, modify the variable `NUM_PROCESSES` in `run.py`:


```python
NUM_PROCESSES = 70  # Set this to the desired number of threads
```

### Single Experiment
For running an individual experiment with specific parameters, use:
```bash
python3 -m TSPHardener.main "[10]" "[5,10,15]" 100 "" --tsp_type "euclidean" --distribution "uniform" --mutation_strategy "scramble"
```

In the above example:
    - City size: 10
    - Control parameter: [5,10,15]
    - 100 generations
    - TSP type: Euclidean
    - Cost distribution: uniform
    - Mutation strategy: scramble

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

# Running

`python3 -m TSPHardener.run` initializes a full experiment given `tsp-formulation.csv` concurently. One can edit or create their own tsp formulation experiment. Each row from the csv file would allocate one thread from the threadpool. To allocate more thread update `NUM_PROCESSES=70` to a new integer in `run.py`
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
- `TSPHardener.utils` : Utility functions for saving experiments, saving partial results, handling continuing from an experiment, and logging.
- `analysis.analysis` :  Independent plotting and statistical scripts analyzing the whole experiment.
- `analysis.util` : Utility function on validating and loading experiment data for analysis.

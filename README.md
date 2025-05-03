# Where Really Hard Traveling Salesman Problems Are

This repository contains the source code for experiments and analyses presented in the thesis Where Really Hard Traveling Salesman Problems Are by Thomas Sijpkens. It extends and builds upon the prior work of Wouter Knibbe's thesis, introducing a broader range of TSP configurations and more varied mutation strategies for both hill climbing and random sampling methods.

To effectively leverage computational resources, each experimental configuration is designed run in parallel, resulting in a total of 620 independent experimental runs.

---

## Running Experiments

### Dependencies

To install all required libraries with `pip install -r requirements.txt`

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

In the example above:
- City size: 10
- Control parameter: [5,10,15]
- 100 generations
- TSP type: Euclidean
- Cost distribution: uniform
- Mutation strategy: scramble

# Project Structure

```plaintext
├── TSPHardener
│   ├── main.py           # Executes a single experiment
│   ├── run.py            # Executes all experiments in parallel
│   ├── core/             # Core logic for the hill-climber
│   └── utils/            # Utility functions for experiment handling
├── analysis/
│   ├── analysis          # Scripts for plotting and statistical analysis
│   └── util              # Utility functions for data validation and loading
├── tsp-formulation.csv   # Definitions of experiment formulations
└── requirements.txt      # Project dependencies
```

Continuing Work

During the hill climbing process, local optima and transition paths are systematically recorded to construct local optima networks. These networks provide detailed insights into the evolutionary trajectories and characteristics of TSP instances, clarifying how different instances evolve under various mutation strategies and random sampling approaches.

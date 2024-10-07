#!/bin/bash

# Name of the Python script to run the experiment
SCRIPT_NAME="experiment.py"

# Arguments for the experiment
CITY_SIZES="[5, 10]"         # List of city sizes to test
VALUE_RANGES="[10, 50]"      # List of value ranges for cost values
MUTATIONS=10                 # Number of mutations to perform
CONTINUATION="[]"            # No continuation for this simple test
GENERATION_TYPE="asymmetric" # Generation type (asymmetric or euclidean)
DISTRIBUTION="uniform"       # Cost distribution type (uniform or lognormal)
MUTATION_STRATEGY="swap"     # Mutation strategy (swap, permute, mutate_random, mutate_symmetric)

# Run the experiment using the defined arguments
echo "Running TSP experiment..."
python3 - $SCRIPT_NAME "$CITY_SIZES" "$VALUE_RANGES" $MUTATIONS "$CONTINUATION" "$GENERATION_TYPE" "$DISTRIBUTION" "$MUTATION_STRATEGY"

echo "Experiment completed."

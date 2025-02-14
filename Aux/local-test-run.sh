#!/bin/bash
# Test 2 configurations with limited resources
CSV_FILE="tsp-formulations.csv"
MAX_CONFIGS=2  # Test first 2 configurations
MUTATIONS=100  # Reduced from 10k for quick testing
PARALLEL_JOBS=2  # Number of parallel processes

# Create test output directory
TEST_DIR="test_runs/$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$TEST_DIR"

# Process CSV in limited mode
tail -n +2 "$CSV_FILE" | head -n $MAX_CONFIGS | while IFS=, read -r city_size tsp_variant cost_distribution mutation_type; do
    {
        # Create unique run directory
        RUN_DIR="${TEST_DIR}/${city_size}_${tsp_variant}_${cost_distribution}_${mutation_type}"
        mkdir -p "$RUN_DIR"
        
        # Parameter conversion
        case $cost_distribution in
            "Uniform") range=1000 ;;
            *) range=1 ;;
        esac
        
        case $mutation_type in
            "Inplace") mutation_strategy="wouter" ;;
            *) mutation_strategy=$(echo "$mutation_type" | tr '[:upper:]' '[:lower:]') ;;
        esac

        echo "=== Running $city_size $tsp_variant $cost_distribution $mutation_type ==="
        python experiment.py \
            "[$city_size]" \
            "[$range]" \
            $MUTATIONS \
            "" \
            --tsp_type "$(echo "$tsp_variant" | tr '[:upper:]' '[:lower:]')" \
            --distribution "$(echo "$cost_distribution" | tr '[:upper:]' '[:lower:]')" \
            --mutation_strategy "$mutation_strategy"
    } > "${RUN_DIR}/output.log" 2>&1 &
    
    # Limit parallel jobs
    if [[ $(jobs -r -p | wc -l) -ge $PARALLEL_JOBS ]]; then
        wait -n
    fi
done

wait
echo "Test completed. Results in $TEST_DIR"
#!/bin/bash
#SBATCH --job-name=tsp_mega_run
#SBATCH --time=120:00:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-23%8  # 24 configs, 8 concurrent jobs
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Configuration mapping
CSV_FILE="/gpfs/home3/tsijpkens/project/TSPHardener/tsp-formulations.csv"
BASE_DIR="/gpfs/home3/tsijpkens/project/TSPHardener/runs"

# Get configuration from CSV using array index
CONFIG_LINE=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+2 {print $0}' $CSV_FILE)
IFS=',' read -r CITY_SIZE TSP_VARIANT COST_DISTRIBUTION MUTATION_TYPE <<< "$CONFIG_LINE"

# Create unique run directory
RUN_DIR="${BASE_DIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${CITY_SIZE}_${TSP_VARIANT}_${COST_DISTRIBUTION}_${MUTATION_TYPE}"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR" || exit 1

# Parameter conversion
{
    [ "$COST_DISTRIBUTION" = "Uniform" ] && RANGE=1000 || RANGE=1
    [ "$MUTATION_TYPE" = "Inplace" ] && MUTATION_STRATEGY="wouter" || MUTATION_STRATEGY="${MUTATION_TYPE,,}"
    TSP_TYPE="${TSP_VARIANT,,}"
    DISTRIBUTION="${COST_DISTRIBUTION,,}"
} 2>/dev/null

# Run experiment with full resource utilization
srun python -u /gpfs/home3/tsijpkens/project/TSPHardener/experiment.py \
    "[$CITY_SIZE]" \
    "[$RANGE]" \
    10000 \
    "" \
    --tsp_type "$TSP_TYPE" \
    --distribution "$DISTRIBUTION" \
    --mutation_strategy "$MUTATION_STRATEGY"

exit 0
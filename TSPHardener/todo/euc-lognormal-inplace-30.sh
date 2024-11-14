#!/bin/bash
#SBATCH --job-name=tsp_experiment
#SBATCH --time=120:00:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

for run in {1..5}; do
    run_dir="/gpfs/home3/tsijpkens/project/TSPHardener/${SLURM_JOB_ID}_run${run}"
    mkdir -p "$run_dir" && cd "$run_dir" || { echo "Failed to create or enter directory $run_dir"; exit 1; }

    srun --exclusive python -u /gpfs/home3/tsijpkens/project/TSPHardener/experiment.py "[30]" "[0.0, 0.2, 0.4, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]" 10000 "" --tsp_type "euclidean" --distribution "lognormal" --mutation_strategy "wouter"
done

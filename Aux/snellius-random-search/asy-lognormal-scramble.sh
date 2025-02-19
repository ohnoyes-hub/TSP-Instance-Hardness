#!/bin/bash
#SBATCH --job-name=tsp_experiment
#SBATCH --time=48:00:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

for run in {1..5}; do
    run_dir="/gpfs/home3/tsijpkens/project/TSPHardener/${SLURM_JOB_ID}_run${run}"
    mkdir -p "$run_dir" && cd "$run_dir" || { echo "Failed to create or enter directory $run_dir"; exit 1; }

    srun python -u /gpfs/home3/tsijpkens/project/TSPHardener/experiment.py "[20]" "[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,]" 10000 "" --tsp_type "asymmetric" --distribution "lognormal" --mutation_strategy "scramble"
done
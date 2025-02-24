#!/bin/bash
#SBATCH --job-name=tsp_experiment
#SBATCH --time=120:00:00
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

for run in {1}; do
    run_dir="/gpfs/home3/tsijpkens/project/TSPHardener/job${SLURM_JOB_ID}_run${run}"
    mkdir -p "$run_dir" && cd "$run_dir" || { echo "Failed to create or enter directory $run_dir"; exit 1; }

    srun python -u /gpfs/home3/tsijpkens/project/TSPHardener/run.py
done

exit 0
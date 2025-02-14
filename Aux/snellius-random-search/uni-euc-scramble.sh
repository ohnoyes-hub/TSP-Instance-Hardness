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

for run in {1}; do
    run_dir="/gpfs/home3/tsijpkens/project/TSPHardener/uni-euc-scramble_${SLURM_JOB_ID}_run${run}"
    mkdir -p "$run_dir" && cd "$run_dir" || { echo "Failed to create or enter directory $run_dir"; exit 1; }

    srun python -u /gpfs/home3/tsijpkens/project/TSPHardener/experiment.py "[20]" "[50]" 10000 "" --tsp_type "euclidean" --distribution "uniform" --mutation_strategy "scramble"
done

exit 0

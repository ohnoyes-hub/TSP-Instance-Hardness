#!/bin/bash
#SBATCH --job-name=tsp_experiment
#SBATCH --time=36:00:00          
#SBATCH --partition=rome       
#SBATCH --nodes=1                
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Repeat the job in 5 different runs
for run in {1..5}; do
    # Create a new directory for this run based on the run number and change to this directory
    run_dir="/gpfs/home3/tsijpkens/project/TSPHardener/${SLURM_JOB_ID}_run${run}"
    mkdir -p "$run_dir"
    cd "$run_dir" || exit

   # Run the experiment
    srun python -u /gpfs/home3/tsijpkens/project/TSPHardener/experiment.py "[30]" "[5,10,15,20,25,30,35,40,45,50]" 10000 "" --tsp_type "asymmetric" --distribution "uniform" --mutation_strategy "wouter"
done


#!/bin/bash
#SBATCH --job-name=tsp_experiment_remaining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=120:00:00 
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tdtsijpkens@gmail.com

cd ~/TSPHardener

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip install -r requirements.txt

python3 run.py
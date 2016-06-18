#!/bin/bash
#SBATCH --job-name=w2wTraining
#SBATCH --partition=slims
#SBATCH -n 20 # Debe de ser un número múltiplo de 20
#SBATCH --output=w2wTraining_%j.out
#SBATCH --error=w2wTraining_%j.err
#SBATCH --mail-user=jotaj.8a@gmail.com
#SBATCH --mail-type=ALL
module load intel impi
module load python/2.7.10

srun python ./TrainWeekToWeek.py
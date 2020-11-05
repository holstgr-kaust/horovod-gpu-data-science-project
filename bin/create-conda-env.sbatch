#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=48G
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --job-name=create-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err


# configure
PROJECT_ROOT=$(realpath "${PROJECT_ROOT:-.}")

# create the conda environment
"${PROJECT_ROOT}/bin/create-conda-env.sh"

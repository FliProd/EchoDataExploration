#!/bin/bash

#SBATCH --job-name=clustering_analysis
#SBATCH --output=clustering_analysis_out.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=10G

source ~/.bashrc
enable_modules
VENV=~/clustering_experiment_1/venv/bin/activate
source $VENV
module load python/3.9.6
module load scipy-stack/2022a
pip install pyclustering

python ~/clustering_experiment_1/run/clustering_experiment.py
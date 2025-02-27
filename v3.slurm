#! /bin/bash
# ====================================
#SBATCH --job-name=A2_SR_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ====================================
# Activate Conda and then the environment.
source ~/software/init-conda
conda activate pytorch

# Use the software here.
python /home/strahinja.radakov1/assignment2/v3/final/A2_SR_v3.py

conda deactivate

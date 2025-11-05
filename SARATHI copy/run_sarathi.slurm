#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --output=sarathi_output_%j.log

echo "Loading modules..."
# IMPORTANT: Change these to the modules that work on your cluster!
module load python/anaconda3-2018.12
module load cuda/11.8

echo "Activating virtual environment..."
source venv/bin/activate

echo "Running SARATHI demo on GPU..."
python main.py
echo "Job complete."
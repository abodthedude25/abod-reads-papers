#!/bin/bash

#SBATCH --job-name=vllm_sim         # Job name
#SBATCH --partition=gpu-a100      # Partition (queue)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=2         # Request 2 CPUs
#SBATCH --mem=16G                 # Request 16GB of memory
#SBATCH --time=00:10:00           # 10-minute time limit
#SBATCH --output=vllm_sim_%j.out  # Standard output log file (%j is the job ID)

# --- Job Steps ---
echo "Job started on $(hostname) at $(date)"

# 1. Navigate to the job directory
#    (Update this to the correct path in your home dir)
cd ~/abod-reads-papers/vllm_pagedattention

# 2. Load the required software
module load python/anaconda3

# 3. Activate your virtual environment
#    (Assuming it's named 'venv' inside your repo)
source ../venv/bin/activate

# 4. Run the python script
echo "Running vLLM PagedAttention simulation..."
python main.py

echo "Job finished at $(date)"
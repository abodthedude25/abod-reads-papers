#!/bin/bash
#SBATCH --job-name=stream_diffusion_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# StreamDiffusion Benchmark Script for HPC Cluster
# This reproduces the paper's benchmark results

echo "=========================================="
echo "StreamDiffusion Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load required modules (adjust for your cluster)
module load cuda/11.8
module load python/3.10
module load gcc/11.2.0

# Activate virtual environment
source venv/bin/activate

# Print system info
echo "System Information:"
echo "===================="
nvidia-smi
echo ""
python --version
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Run benchmarks
echo "Starting benchmarks..."
echo ""

# 1. Component ablation
echo "1. Component Ablation Study"
echo "==========================="
python benchmarks/benchmark_components.py
echo ""

# 2. Throughput benchmark
echo "2. Throughput Benchmark (reproducing Table 1)"
echo "=============================================="
python benchmarks/benchmark_throughput.py --mode comprehensive
echo ""

# 3. Quick validation
echo "3. Quick Validation"
echo "==================="
python benchmarks/benchmark_throughput.py --mode quick
echo ""

# 4. Example generation
echo "4. Example Generation"
echo "====================="
python examples/text_to_image.py --mode basic
echo ""

echo "=========================================="
echo "Benchmark Complete!"
echo "Results saved to: benchmark_$SLURM_JOB_ID.out"
echo "=========================================="
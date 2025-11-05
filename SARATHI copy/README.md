1. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked PrefillsThis project is a functional implementation of the core concepts from the paper SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.It demonstrates the two key breakthroughs of the paper:Chunked Prefills: Splitting a large prompt (e.g., 512 tokens) into smaller "chunks" (e.g., two 256-token chunks) produces the exact same output, proving it's a "lossless" optimization.Decode-Maximal Batching ("Piggybacking"): The slow, memory-bound FFN (linear layer) computations for a batch of decode tokens can be "piggybacked" onto the fast, compute-bound FFN computations of a prefill chunk. This makes the decode operations dramatically faster.The Core ProblemLLM inference is inefficient because its two phases have opposite performance characteristics:Inefficient Decodes (Memory-Bound): Generating one token at a time is a "matrix-vector" operation. The GPU spends ~99% of its time slowly loading massive model weights from VRAM and ~1% of its time doing math. This is incredibly wasteful.Efficient Prefills (Compute-Bound): Processing a prompt (e.g., 256+ tokens) is a "matrix-matrix" operation. The GPU loads the weights once and does a massive amount of math, achieving high utilization.SARATHI's SolutionSARATHI stops running inefficient decode-only batches. Instead, it creates hybrid batches that have the best of both worlds.Chunked Prefills: A large prompt (e.g., 2048 tokens) is broken into small, uniform pieces (e.g., 8 chunks of 256 tokens). This:Creates 8 opportunities for piggybacking (instead of just 1).Solves pipeline bubbles by making all prefill jobs a uniform size.Decode-Maximal Batching (Piggybacking): The scheduler creates a hybrid batch with 1 prefill chunk and N decode tokens.The Attention layers are run separately.The FFN (Linear) layers are fused into one big operation.The prefill chunk "pays" the high cost of loading the model weights, and the decode tokens get an almost-free ride, as their computations are just added to the already-happening compute-bound operation.How to Run This DemoThis project is designed to be run on a GPU (NVIDIA) to see the performance effect.1. Setup (Local or on HPC)# Clone the main repo (if you haven't)
git clone [https://github.com/YourUsername/abod-reads-papers.git](https://github.com/YourUsername/abod-reads-papers.git)
cd abod-reads-papers/sarathi

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install the GPU-enabled version of PyTorch
# (Get the correct command for your CUDA version from pytorch.org)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
2. Run on the ARC Cluster (Recommended)You must request a GPU from the SLURM scheduler.Option A: Interactive Job (for testing)# 1. Request an A100 GPU for 10 minutes
salloc --mem=8G -t 00:10:00 -p gpu-a100 --gres=gpu:1

# 2. Once on the GPU node (e.g., 'fg3'), load modules and activate env
module load python/anaconda3-2018.12  # Or your preferred python
module load cuda/11.8                  # Or your installed CUDA version
source venv/bin/activate

# 3. Run the script!
python main.py
Option B: Batch Job (the "pro" way)The included run_sarathi.slurm script is pre-configured.# Submit the job from the login node
sbatch run_sarathi.slurm

# Check the output once it's done
cat sarathi_output_*.log
Understanding the ResultsThe script runs two tests.Test 1: Correctness CheckThis test proves that processing a 512-token prompt in one go is mathematically identical (within bfloat16 precision) to processing it in two 256-token chunks.--- Test 1: Chunked Prefill Correctness Check ---
...
[SUCCESS] Chunked prefill output matches full prefill output!
Test 2: Performance "Aha!" MomentThis is the core of the paper. It compares the baseline (running prefill and decode separately) to SARATHI's fused method.Example GPU Output (NVIDIA A100):--- Test 2: SARATHI Performance Demo (Piggybacking) ---
[Baseline] Time for 1x Prefill Chunk (S1):  0.9982 ms
[Baseline] Time for 1x Decode Batch (S2):   1.2165 ms
[SARATHI]  Time for 1x Fused Batch (S3):   1.8722 ms

--- Analysis ---
Baseline Total (S1 + S2):    2.2148 ms
SARATHI Total (S3):          1.8722 ms
Total Speedup (Baseline / SARATHI): 1.18x

--- The 'Aha!' Moment (Decode Cost) ---
Full cost of decodes (S2):             1.2165 ms
Marginal cost of decodes (S3 - S1):  0.8740 ms

==> Decode Piggybacking Speedup: 1.39x <==
Analysis:Baseline: Running the jobs separately would take 0.99 (S1) + 1.21 (S2) = 2.21 ms.SARATHI: Running them fused takes only 1.87 ms.The "Aha!" Moment: The full cost of the slow decode batch was 1.21 ms. By fusing it, the marginal cost (the extra time added to the prefill chunk) was only 1.87 (S3) - 0.99 (S1) = 0.87 ms.This makes the decode operation itself 1.39x faster, proving the piggybacking concept works.
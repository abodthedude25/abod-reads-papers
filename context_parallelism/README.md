# 3\. Context Parallelism

  * **Paper:** [Context Parallelism for Scalable Million-Token Inference (arXiv:2411.01783)](https://arxiv.org/abs/2411.01783)
  * **Code:**
      * **Simulation:** `main.py`, `algorithms.py`, `heuristic.py`
      * **Benchmark:** `main_performance.py`, `distributed_algorithms.py`, `dist_utils.py`
      * **Shared:** `attention.py`, `sharding.py`

-----

### 🎯 Core Concept

This paper solves the two primary bottlenecks for million-token inference:

1.  **Prefill:** The massive O(T²) compute for the initial prompt.
2.  **Decode:** The memory capacity required to *store* the KV cache for T=1M.

The solution is **Context Parallelism (CP)**, which shards the input *sequence* (the context) across multiple GPUs (CP ranks). This repository contains two implementations:

1.  A **logical simulation** to prove the algorithms are numerically correct.
2.  A **distributed benchmark** to prove the algorithms are faster on real multi-GPU hardware.

The core algorithms demonstrated are:

  * **Ring Pass-KV (Algorithm 2):** Each rank holds its **Q** stationary and passes its **K/V** tensors around a ring. Used for full prefill.
  * **Ring Pass-Q (Algorithm 3):** Each rank holds its **K/V** stationary and passes its **Q** tensor around a ring. Used for partial prefill.
  * **Ring Pass-Q Decode (Algorithm 4):** A variant of Pass-Q for auto-regressive decoding (`T=1`) that balances KV cache load.

-----

### 💡 Code Implementation

This project is split into two distinct parts:

#### 1\. Logical Simulation (Correctness)

This is a single-process simulation that verifies the *mathematical correctness* of the distributed algorithms. It proves that the sharded, ring-based attention produces a bit-for-bit identical result to a standard, non-parallel attention.

  * `main.py`: The main harness that runs the simulation and compares the results to a ground truth.
  * `algorithms.py`: A high-level *simulation* of the algorithms. It fakes the ring communication by rotating lists in a single process.
  * `heuristic.py`: Implements the `heuristic_select_mode` function (Algorithm 5) to decide between pass-KV and pass-Q.

#### 2\. Distributed Performance Benchmark (Speed)

This is a true multi-GPU implementation that uses `torch.distributed` to benchmark the *real-world speedup* of Context Parallelism. It launches multiple processes and sends tensors between GPUs using the NCCL backend.

  * `main_performance.py`: The main script to launch the multi-GPU benchmark. It compares the wall-clock time of a single-GPU ground truth against the N-GPU distributed version.
  * `distributed_algorithms.py`: The *real* implementation of the Ring Pass-KV algorithm. It uses blocking `dist.send`/`dist.recv` and handles low-level CUDA bugs (like `long` tensor casting) for a robust, correct implementation.
  * `dist_utils.py`: A helper file to manage setting up and tearing down the `torch.distributed` process group.

#### Shared Modules

  * `attention.py`: Contains the core compute kernels: `simple_attention` and `merge_attention_outputs` (Appendix B, Eq. 4). This math is shared by both implementations.
  * `sharding.py`: Implements the `get_load_balanced_shards` logic (Section 3.5.1), also shared by both.

-----

### 🚀 How to Run

There are two ways to run this code, corresponding to the two implementations.

#### 1\. Running the Logical Simulation

This is a simple, single-process test to verify the math. It can be run on any machine with PyTorch (even a laptop CPU).

```bash
# 1. Make sure you have PyTorch installed
pip install torch

# 2. Run the main simulation script
python main.py
```

**Expected Output (Simulation):**

```
--- ⚙️ System Parameters ---
CP Ranks (N): 4
Seq Len (T): 64
Head Dim (D_H): 32

--- 📊 Ground Truth (Single Rank) ---
Running standard causal attention...
  O_truth shape: torch.Size([1, 64, 32])

--- 🚀 Simulating Distributed Prefill (P=0) ---
Sharding T=64 into 4 ranks...
  Rank 0 indices: [ 0  1  2  3  4  5  6  7 56 57 58 59 60 61 62 63]
  ...

--- 🚀 Running Ring Pass-KV (Algorithm 2) ---
  Verifying...
✅ SUCCESS: Pass-KV output matches ground truth.

--- 🚀 Running Ring Pass-Q (Algorithm 3) ---
  Verifying...
✅ SUCCESS: Pass-Q output matches ground truth.

--- 🤔 Heuristic Demonstration (Algorithm 5) ---
  ...
  Miss Rate:  1.0% (T= 1280, P=126720) -> Choose: pass-Q (Q is tiny, KV is huge)
  Miss Rate: 15.0% (T=19200, P=108800) -> Choose: pass-KV (T is large enough to hide KV comm)

--- 🔄 Simulating Decode (Algorithm 4) ---
  ...
✅ SUCCESS: Decode simulation complete. KV cache is balanced.
```

#### 2\. Running the Performance Benchmark

This requires a multi-GPU environment, such as the ARC cluster. This test will prove the **speedup** from Context Parallelism.

```bash
# 1. Log in to ARC
ssh your_username@arc.ucalgary.ca

# 2. Request 2 GPUs on a single node (e.g., in the gpu-a100 partition)
# We request 64GB of RAM to be safe.
salloc --partition=gpu-a100 --gres=gpu:2 --nodes=1 --mem=64G --time=00:10:00

# 3. Once on the compute node (e.g., fg6), navigate to your code
cd ~/abod-reads-papers/context_parallelism

# 4. Load Python and activate your environment
module load python/anaconda3
source venv/bin/activate

# 5. (If not already done) Install PyTorch
pip install torch

# 6. Run the performance benchmark script!
python main_performance.py
```

**Expected Output (Benchmark):**

```
Starting performance test with 2 GPUs for T=16384, B=4...
Rank 0 initialized on GPU 0
Rank 1 initialized on GPU 1

--- 📊 Ground Truth (Single GPU) ---
  ✅ Ground Truth Time: 122.47 ms

--- 🚀 Distributed (Context Parallelism, 2 GPUs) ---

  📊 Detailed Timing Breakdown (Rank 0):
     Total compute:  73.23ms ( 74.3%)
     Total comm:     25.37ms ( 25.7%)
     ─────────────────────────────────
     Total time:     98.60ms
     ✅ Compute is the bottleneck (as expected).
  ✅ Distributed Time: 99.22 ms

--- ✅ Verification (on Rank 0) ---
  ✅ SUCCESS: Distributed output matches ground truth.
```
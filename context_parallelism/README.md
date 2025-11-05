# 3. Context Parallelism

* **Paper:** [Context Parallelism for Scalable Million-Token Inference (arXiv:2411.01783)](https://arxiv.org/abs/2411.01783)
* **Code:** `main.py`, `algorithms.py`, `attention.py`, `sharding.py`, `heuristic.py`

### 🎯 Core Concept

This paper solves the two primary bottlenecks for million-token inference:
1.  **Prefill:** The massive O(T²) compute for the initial prompt.
2.  **Decode:** The memory capacity required to *store* the KV cache for T=1M.

The solution is **Context Parallelism (CP)**, which shards the input *sequence* (the context) across multiple GPUs (CP ranks). This implementation demonstrates all three core algorithms presented in the paper, proving they are numerically identical to a standard, non-parallel attention mechanism.

1.  **Ring Pass-KV (Algorithm 2):** Each rank holds its **Q** stationary and passes its **K/V** tensors around a ring.
    * **Use Case:** Full prefill (P=0), especially with GQA (where K/V tensors are small).
    * **File:** `algorithms.py`

2.  **Ring Pass-Q (Algorithm 3):** Each rank holds its **K/V** stationary and passes its **Q** tensor around a ring.
    * **Use Case:** Partial prefill (multi-turn chat) where you have a huge cached KV (`P`) and a small new prompt (`T`). Since `T << P`, passing Q is much cheaper.
    * **File:** `algorithms.py`

3.  **Ring Pass-Q Decode (Algorithm 4):** A variant of Pass-Q for auto-regressive decoding (`T=1`). It uses round-robin sharding to distribute new KV cache entries, balancing memory load across all ranks.
    * **Use Case:** Generating response tokens one-by-one.
    * **File:** `algorithms.py`

### 💡 Code Implementation

The logic is split into multiple files for clarity:

* `attention.py`: Contains the core compute kernels: `simple_attention` and the numerically-stable `merge_attention_outputs` (Appendix B, Eq. 4).
* `sharding.py`: Implements the `get_load_balanced_shards` logic for prefill (Section 3.5.1).
* `algorithms.py`: Implements the high-level logic for Algorithms 2, 3, and 4.
* `heuristic.py`: Implements the `heuristic_select_mode` function (Algorithm 5) to dynamically choose between pass-KV and pass-Q.
* `main.py`: The main simulation harness. It sets up a `CPNetwork` simulation, runs all algorithms, verifies their numerical correctness against a ground truth, and demonstrates the heuristic and decode logic.

### 🚀 How to Run

No special libraries are needed beyond PyTorch.

```bash
# 1. Make sure you have PyTorch installed
pip install torch

# 2. Run the main verification script
python main.py
```

You will see output verifying that both Pass-KV and Pass-Q prefill are lossless, followed by a demonstration of the heuristic and the decode simulation.

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
  Rank 0 indices: [ 0,  1,  2,  3,  4,  5,  6,  7, 56, 57, 58, 59, 60, 61, 62, 63]
  Rank 1 indices: [ 8,  9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55]
  Rank 2 indices: [16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47]
  Rank 3 indices: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

--- 🚀 Running Ring Pass-KV (Algorithm 2) ---
  Verifying...
✅ SUCCESS: Pass-KV output matches ground truth.

--- 🚀 Running Ring Pass-Q (Algorithm 3) ---
  Verifying...
✅ SUCCESS: Pass-Q output matches ground truth.

--- 🤔 Heuristic Demonstration (Algorithm 5) ---
Running heuristic for different KV cache hit rates (T_new + P_cache = 128k)
  Miss Rate:  1.0% (T=1280, P=126720) -> Choose: pass-Q (Q is tiny, KV is huge)
  Miss Rate:  5.0% (T=6400, P=121600) -> Choose: pass-Q
  Miss Rate: 10.0% (T=12800, P=115200) -> Choose: pass-Q
  Miss Rate: 12.5% (T=16000, P=112000) -> Choose: pass-Q
  Miss Rate: 15.0% (T=19200, P=108800) -> Choose: pass-KV (T is large enough to hide KV comm)
  Miss Rate: 50.0% (T=64000, P=64000) -> Choose: pass-KV
  Miss Rate: 100.0% (T=128000, P=0)  -> Choose: pass-KV (Full prefill)

--- 🔄 Simulating Decode (Algorithm 4) ---
Prefilling network with 64 tokens...
Network KV cache state: [R0: 16, R1: 16, R2: 16, R3: 16] (Total: 64)

Running 4 decode steps (T=1)...
  Decode Step 0: Q assigned to Rank 0.
    Network KV cache state: [R0: 17, R1: 16, R2: 16, R3: 16] (Total: 65)
  Decode Step 1: Q assigned to Rank 1.
    Network KV cache state: [R0: 17, R1: 17, R2: 16, R3: 16] (Total: 66)
  Decode Step 2: Q assigned to Rank 2.
    Network KV cache state: [R0: 17, R1: 17, R2: 17, R3: 16] (Total: 67)
  Decode Step 3: Q assigned to Rank 3.
    Network KV cache state: [R0: 17, R1: 17, R2: 17, R3: 17] (Total: 68)
✅ SUCCESS: Decode simulation complete. KV cache is balanced.
```
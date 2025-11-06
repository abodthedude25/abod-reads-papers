# 2. vLLM / PagedAttention

* **Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
* **Code:** `main.py`, `kv_cache_manager.py`, `paged_attention.py`, `sequence.py`

### 🎯 Core Concept

This paper solves the single biggest bottleneck in LLM serving: **memory waste**.

**The Problem:**
When serving many users, the GPU memory is dominated by the **KV Cache** (the "memory" of past tokens). In previous systems (like FasterTransformer), the entire KV Cache for one request had to be stored in a **single, contiguous block of memory**.

Because a request's output length is unpredictable, the system had to "guess" and pre-allocate a huge block (e.g., 2048 tokens). This resulted in massive memory waste from:
1.  **Internal Fragmentation:** A 2048-token block is reserved, but the user's request only generates 50 tokens. **97% of the memory is wasted.**
2.  **External Fragmentation:** The GPU memory becomes "Swiss cheese," with many small, unusable free chunks.
3.  **No Sharing:** If you request 3 outputs from one prompt, the system had to *duplicate* the prompt's KV cache 3 times, wasting even more memory.

The paper showed this waste can be **60-80%** of the total KV Cache memory.

**The Solution: PagedAttention**
The solution is to borrow a 40-year-old idea from Operating Systems: **Virtual Memory and Paging**.

1.  **No More Contiguous Memory:** The KV cache is broken into small, fixed-size **"blocks"**.
2.  **Virtual-to-Physical Mapping:** A request's *logical* sequence of tokens is mapped to *physical* blocks that are scattered all over the GPU memory.
3.  **On-Demand Allocation:** Memory is allocated one block at a time, only when it's needed. This **eliminates** internal and external fragmentation.
4.  **Copy-on-Write (CoW):** This is the superpower. For parallel sampling (3 outputs, 1 prompt), all 3 sequences can *point to the same physical blocks* for the prompt. The data is only stored *once*.

### 💡 Code Implementation

This project is a **memory usage simulation**, not a latency benchmark. It runs on a single GPU and proves the memory-saving claims of the paper.

* `kv_cache_manager.py`: Contains two classes:
    * `BaselineCacheManager`: Simulates the old "contiguous" allocator, which suffers from internal fragmentation.
    * `PagedKVCacheManager`: A full simulation of the vLLM memory manager, including a "free list" of blocks, block tables, reference counting, and Copy-on-Write.
* `sequence.py`: A helper class to represent a single sequence (request) and its associated `BlockTable`.
* `paged_attention.py`: Contains a `paged_attention_kernel` function. This is a *simulation* of the custom CUDA kernel. It "gathers" the K/V data from the scattered physical blocks (using the sequence's block table) into a contiguous tensor before running attention.
* `attention.py`: The exact same, bit-identical attention kernel from our Context Parallelism implementation.
* `main.py`: The main showcase script. It runs four scenarios to demonstrate every benefit mentioned in the paper.
* `run_vllm_sim.slurm`: The Slurm script to run this simulation on an ARC GPU node.

### 🚀 How to Run

This simulation runs on a single GPU.

```bash
# 1. Log in to ARC
ssh your_username@arc.ucalgary.ca

# 2. Request a single GPU (we need it for the tensor allocation)
salloc --partition=gpu-a100 --gres=gpu:1 --mem=16G --time=00:10:00

# 3. Once on the compute node, navigate to your code
cd ~/abod-reads-papers/vllm_pagedattention

# 4. Load Python and activate your environment
module load python/anaconda3
source venv/bin/activate

# 5. (If not already done) Install PyTorch
pip install torch

# 6. Run the main simulation!
python main.py
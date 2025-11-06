
# abod-reads-papers

This repository is my personal collection of notes and functional implementations of recent, high-impact AI/ML papers, with a special focus on performance, systems, and inference optimization.

The goal is to move beyond just *reading* the paper and to *prove* the core concepts by implementing them in code.

## Papers Implemented

Each paper has its own self-contained directory, which includes:

  * `README.md`: A detailed breakdown of the paper's core problem, its solution, and how to run the code.
  * Source Code: A minimal, functional implementation of the paper's core idea.
  * `*.slurm` script: (If applicable) A batch script for running the performance test on an HPC cluster like ARC.

-----

### 1\. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

  * **Paper:** [arxiv.org/abs/2308.16369](https://arxiv.org/abs/2308.16369)
  * **Core Concept:** Solves the inefficient, memory-bound decode phase of LLM inference. It does this by "piggybacking" a batch of decode tokens onto a compute-bound "prefill chunk," fusing their FFN (linear) layers into a single, efficient operation.
  * **Status:** Complete. The implementation proves the "piggybacking" speedup on a real GPU.

-----

### 2\. vLLM / PagedAttention 

  * **Paper:** [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
  * **Core Concept:** Solves the massive memory fragmentation problem in LLM serving by bringing Operating Systems concepts (virtual memory and paging) to the GPU's KV Cache. This is the core technology of the vLLM library.
  * **Status:** Complete.

-----

### 3\. Context Parallelism 

  * **Paper:** [arxiv.org/abs/2411.01783](https://arxiv.org/abs/2411.01783)
  * **Core Concept:** Solves the O(T²) compute bottleneck for million-token prefill by introducing "pass-Q" and "pass-KV" ring attention variants, which are dynamically selected by a heuristic.
  * **Status:** Complete.

## How to Use This Repo

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/abod-reads-papers.git
    ```
2.  Each paper is in its own directory. `cd` into the one you want to explore:
    ```bash
    cd sarathi
    ```
3.  Follow the instructions in that directory's specific `README.md` file.

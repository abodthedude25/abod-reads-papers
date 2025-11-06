import torch
import random
import math
from kv_cache_manager import BaselineCacheManager, PagedKVCacheManager

# --- CONFIGURATION ---
NUM_BLOCKS = 1024         # Total physical blocks on our 'GPU'
BLOCK_SIZE = 16           # Number of tokens per block
MAX_SEQ_LEN = 2048        # Max context length for baseline

# Model params
NUM_HEADS = 12
HEAD_SIZE = 64
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_header(title):
    print("\n" + "---" * 10)
    print(f"--- {title} ---")
    print("---" * 10)

def main():
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This simulation is for memory logic, not speed.")

    # Calculate total simulated memory
    bytes_per_token = 2 * NUM_HEADS * HEAD_SIZE * (2 if DTYPE == torch.float16 else 4)
    total_mem_gb = (NUM_BLOCKS * BLOCK_SIZE * bytes_per_token) / (1024**3)

    print_header("CONFIGURATION")
    print(f"Total Physical Blocks: {NUM_BLOCKS}")
    print(f"Block Size (Tokens): {BLOCK_SIZE}")
    print(f"Max Seq Len (Baseline): {MAX_SEQ_LEN}")
    print(f"Simulated GPU Memory: {total_mem_gb:.2f} GB")


    # --- SCENARIO 1: The Problem (Internal Fragmentation) ---
    print_header("SCENARIO 1: The Problem (Baseline)")
    print(f"Simulating 100 requests with random lengths (10 to 100 tokens)")
    print(f"Max sequence length is {MAX_SEQ_LEN}. All allocations are {MAX_SEQ_LEN}.")
    
    baseline_manager = BaselineCacheManager(max_seq_len=MAX_SEQ_LEN)
    
    # Generate the same workload for both managers
    random.seed(42)
    workload = [random.randint(10, 100) for _ in range(100)]
    
    for num_tokens in workload:
        baseline_manager.alloc(num_tokens)
    
    baseline_manager.report_waste()

    # --- SCENARIO 2: The PagedAttention Solution ---
    print_header("SCENARIO 2: The PagedAttention Solution")
    print("Simulating the *same* 100 requests with PagedAttention")

    paged_manager = PagedKVCacheManager(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, DTYPE, DEVICE
    )

    for i, num_tokens in enumerate(workload):
        paged_manager.alloc_sequence(seq_id=i, num_prompt_tokens=num_tokens)
    
    paged_manager.report_waste()
    
    # Clean up for next scenario
    for i in range(len(workload)):
        paged_manager.free_sequence(seq_id=i)

    # --- SCENARIO 3: Copy-on-Write (Parallel Sampling) ---
    print_header("SCENARIO 3: Copy-on-Write (Parallel Sampling)")
    print("Simulating 1 prompt (30 tokens) with 10 parallel outputs (20 new tokens each)")
    
    PROMPT_LEN = 30
    GEN_LEN = 20
    NUM_OUTPUTS = 10
    
    # Baseline: 10 requests, each is (30 + 20) tokens long
    baseline_total_tokens = (PROMPT_LEN + GEN_LEN) * NUM_OUTPUTS
    baseline_total_alloc = MAX_SEQ_LEN * NUM_OUTPUTS
    baseline_waste = (baseline_total_alloc - baseline_total_tokens) / baseline_total_alloc
    
    print(f"  - Baseline (no sharing):")
    print(f"    - Total tokens stored: {baseline_total_tokens}")
    print(f"    - Total tokens ALLOCATED: {baseline_total_alloc} (waste: {baseline_waste:.1%})")

    # PagedAttention with CoW
    # 1. Allocate the prompt
    paged_manager.alloc_sequence(seq_id="prompt", num_prompt_tokens=PROMPT_LEN)
    
    # 2. Fork 10 outputs from the prompt
    for i in range(NUM_OUTPUTS):
        paged_manager.fork(parent_seq_id="prompt", new_seq_id=i)
    
    # 3. "Generate" 20 new tokens for each output
    for i in range(NUM_OUTPUTS):
        for _ in range(GEN_LEN):
            paged_manager.append(seq_id=i)
            
    # We can now free the original prompt seq, the children hold the refs
    paged_manager.free_sequence("prompt")

    print(f"  - PagedAttention (with CoW):")
    
    # --- FIX: Manual, correct calculation for CoW scenarios ---
    total_unique_tokens = PROMPT_LEN + (GEN_LEN * NUM_OUTPUTS)
    paged_blocks = paged_manager.get_total_blocks_used()
    total_allocated_slots = paged_blocks * BLOCK_SIZE
    real_waste = (total_allocated_slots - total_unique_tokens)
    real_waste_percent = (real_waste / total_allocated_slots) * 100

    print(f"    - Total *unique* tokens stored: {total_unique_tokens}")
    print(f"    - Total blocks used: {paged_blocks}")
    print(f"    - Total tokens ALLOCATED: {total_allocated_slots} ({paged_blocks} blocks * {BLOCK_SIZE} tokens)")
    print(f"    - ✅ WASTED MEMORY (Padding): {real_waste_percent:.1f}%")
    
    # Calculate saving
    baseline_blocks_no_sharing = math.ceil((PROMPT_LEN + GEN_LEN) / BLOCK_SIZE) * NUM_OUTPUTS
    saving = (baseline_blocks_no_sharing - paged_blocks) / baseline_blocks_no_sharing
    
    print(f"  - ✅ MEMORY SAVING vs. Paged-No-Sharing (in blocks): {saving:.1%}")
    # --- END FIX ---

    # Clean up
    for i in range(NUM_OUTPUTS):
        paged_manager.free_sequence(i)

    # --- SCENARIO 4: Copy-on-Write (Beam Search) ---
    print_header("SCENARIO 4: Copy-on-Write (Beam Search)")
    print("Simulating a 3-step beam search (width=3)")
    
    BEAM_WIDTH = 3
    prompt_len = 10
    prompt_blocks = math.ceil(prompt_len / BLOCK_SIZE)
    
    # Step 0: Prefill prompt
    paged_manager.alloc_sequence(seq_id="prompt", num_prompt_tokens=prompt_len)
    print(f"  - Step 0: Prefill prompt (10 tokens). ({prompt_blocks} blocks used)")
    paged_manager.print_ref_counts()

    # Step 1: Fork 3 beams from the parent
    for i in range(BEAM_WIDTH):
        paged_manager.fork("prompt", i)
    # The parent prompt is now just a read-only template
    paged_manager.free_sequence("prompt") 
    print(f"  - Step 1: Fork {BEAM_WIDTH} beams. All beams share the {prompt_blocks} prompt blocks.")
    paged_manager.print_ref_counts()

    # Step 2: All beams generate 1 new token. This triggers CoW.
    for i in range(BEAM_WIDTH):
        paged_manager.append(i)
    print(f"  - Step 2: All beams generate 1 new token. (CoW on last block)")
    print("    - All beams now have a unique last block. (Early blocks still shared)")
    paged_manager.print_ref_counts()

    # Step 3: Beam 0 "dies" (is pruned)
    paged_manager.free_sequence(0)
    print(f"  - Step 3: Beam 0 dies.")
    print("    - Ref counts for its blocks are decremented.")
    paged_manager.print_ref_counts()
    print(f"  - ✅ Beam search simulation complete.")
    
    # Clean up remaining beams
    for i in range(1, BEAM_WIDTH):
        paged_manager.free_sequence(i)

    # --- SCENARIO 5: Copy-on-Write (Shared System Prefix) ---
    print_header("SCENARIO 5: Copy-on-Write (Shared System Prefix)")
    print("Simulating 50 users with the same 100-token system prompt")
    
    PREFIX_LEN = 100
    NUM_USERS = 50

    # 1. "Cache" the prefix
    paged_manager.alloc_sequence("system_prefix", PREFIX_LEN)
    prefix_blocks_count = len(paged_manager.sequences["system_prefix"].get_block_table())
    print(f"  - Caching 1 prefix... ({prefix_blocks_count} blocks)")

    # 2. Fork 50 users from it
    for i in range(NUM_USERS):
        paged_manager.fork("system_prefix", f"user_{i}")
        # In a real system, you'd now append user's prompt
    
    # The original prefix can be freed, the users hold the refs
    paged_manager.free_sequence("system_prefix")
    
    print(f"  - 50 new requests are 'forked' from the cached prefix")
    
    total_blocks_used = paged_manager.get_total_blocks_used()
    mem_used_mb = (total_blocks_used * BLOCK_SIZE * bytes_per_token) / (1024**2)
    print(f"  - Total blocks used by all 50 requests: {total_blocks_used} (shared)")
    print(f"  - Total memory used: {mem_used_mb:.2f} MB")

    # Compare to baseline
    baseline_mem_mb = (NUM_USERS * PREFIX_LEN * bytes_per_token) / (1024**2)
    print(f"  - (Memory if not shared: {baseline_mem_mb:.2f} MB)")
    print(f"  - ✅ Prefix Sharing Saved: {(1 - mem_used_mb/baseline_mem_mb):.1%}")
    
    # Clean up
    for i in range(NUM_USERS):
        paged_manager.free_sequence(f"user_{i}")

if __name__ == "__main__":
    main()
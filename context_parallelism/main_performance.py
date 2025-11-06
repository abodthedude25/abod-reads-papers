import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

from sharding import get_load_balanced_shards
from attention import simple_attention
from dist_utils import setup, cleanup
from distributed_algorithms import run_ring_pass_kv_distributed

# ---
# Benchmark Functions
# ---

def benchmark_ground_truth(rank, Q_full, K_full, V_full):
    """
    Runs the non-parallel ground truth on a single GPU.
    Tensors are pre-allocated and passed in.
    """
    if rank != 0:
        return None  # Only rank 0 runs this
        
    print("\n--- 📊 Ground Truth (Single GPU) ---")
    
    # Tensors are already on the correct device (cuda:0)
    device = Q_full.device
    
    # --- Create Causal Mask ---
    T = Q_full.shape[1]
    try:
        causal_mask_full = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).unsqueeze(0)
    except torch.cuda.OutOfMemoryError:
        print(f"  ❌ FAILED: OOM on single GPU with T={T} (causal mask).")
        return None
    except Exception as e:
        print(f"  ❌ FAILED: Error creating mask: {e}")
        return None

    # Warmup
    try:
        for _ in range(3):
            _ = simple_attention(Q_full, K_full, V_full, causal_mask=causal_mask_full)
    except Exception as e:
        print(f"  ❌ FAILED: Error during warmup: {e}")
        return None
        
    # Timing with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()

    O_truth, _ = simple_attention(Q_full, K_full, V_full, causal_mask=causal_mask_full)

    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"  ✅ Ground Truth Time: {elapsed_ms:.2f} ms")
    return O_truth


def benchmark_distributed(rank, world_size, T, D_H, B, dtype, Q_full_cpu, K_full_cpu, V_full_cpu):
    """Runs the distributed Ring Pass-KV on N GPUs."""
    
    # Ensure all processes are in sync before we start
    dist.barrier()
    
    if rank == 0:
        print(f"\n--- 🚀 Distributed (Context Parallelism, {world_size} GPUs) ---")
        
    device = torch.device(f"cuda:{rank}")
    
    # 1. Get sharding info and create local shards
    shard_indices_list = get_load_balanced_shards(T, world_size)
    my_indices = shard_indices_list[rank]
    my_indices_tensor = torch.tensor(my_indices, dtype=torch.long, device=device)
    
    # Slice the full tensors to get this rank's shards (from CPU tensor)
    q_shard = Q_full_cpu[:, my_indices, :].to(device)
    k_shard = K_full_cpu[:, my_indices, :].to(device)
    v_shard = V_full_cpu[:, my_indices, :].to(device)
    
    # Warmup
    for _ in range(3):
        _ = run_ring_pass_kv_distributed(rank, world_size, q_shard, k_shard, v_shard, my_indices_tensor, profile=False)

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    dist.barrier()  # Sync all processes before starting timer
    
    start_event.record()
    
    # This is the function we're testing. Pass profile=True
    O_shard = run_ring_pass_kv_distributed(rank, world_size, q_shard, k_shard, v_shard, my_indices_tensor, profile=True)
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    
    # Each rank gathers its time. We'll use the max.
    times = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(times, torch.tensor(elapsed_ms, device=device))
    
    if rank == 0:
        max_time = max(t.item() for t in times)
        print(f"  ✅ Distributed Time: {max_time:.2f} ms")
        
    # Return the shard and indices for verification
    return O_shard, my_indices


def main_worker(rank, world_size, T, D_H, B, dtype):
    """The main function for each spawned process."""
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        # --- Create Full Tensors (on CPU first to avoid OOM on rank 0) ---
        torch.manual_seed(42)
        Q_full_cpu = torch.randn(B, T, D_H, dtype=dtype)
        K_full_cpu = torch.randn(B, T, D_H, dtype=dtype)
        V_full_cpu = torch.randn(B, T, D_H, dtype=dtype)
        
        # Move rank 0's full data to its GPU for the ground truth test
        Q_full_gpu0 = Q_full_cpu.to(device) if rank == 0 else None
        K_full_gpu0 = K_full_cpu.to(device) if rank == 0 else None
        V_full_gpu0 = V_full_cpu.to(device) if rank == 0 else None

        # --- Run Benchmarks ---
        O_truth = benchmark_ground_truth(rank, Q_full_gpu0, K_full_gpu0, V_full_gpu0)
        
        O_shard_dist, my_indices = benchmark_distributed(
            rank, world_size, T, D_H, B, dtype,
            Q_full_cpu, K_full_cpu, V_full_cpu
        )
        
        # --- Verification ---
        if O_truth is not None:  # Only rank 0 can verify
            print("\n--- ✅ Verification (on Rank 0) ---")
            O_truth_shard = O_truth[:, my_indices, :]
            if torch.allclose(O_truth_shard, O_shard_dist, atol=1e-4):
                print("  ✅ SUCCESS: Distributed output matches ground truth.")
            else:
                diff = (O_truth_shard - O_shard_dist).abs().max()
                print(f"  ❌ FAILURE: Output mismatch. Max diff: {diff}")
    
    except Exception as e:
        print(f"--- ❌ ERROR IN RANK {rank} ---")
        print(e)
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup()

if __name__ == "__main__":
    # --- System Parameters ---
    N_GPUS = 2         # Set this to the number of GPUs you want to test (e.g., 2 or 4)
    
    # Start with 8192, then try 16384 or 32768.
    T_BENCH = 16384
    
    D_HEAD = 64
    B = 4
    dtype = torch.float32

    # Check if we have enough GPUs
    if N_GPUS > torch.cuda.device_count():
        print(f"Error: Requested N_GPUS={N_GPUS}, but only {torch.cuda.device_count()} are available.")
        exit(1)
        
    print(f"Starting performance test with {N_GPUS} GPUs for T={T_BENCH}, B={B}...")
    
    # `mp.spawn` launches `N_GPUS` copies of the `main_worker` function
    mp.spawn(main_worker,
             args=(N_GPUS, T_BENCH, D_HEAD, B, dtype),
             nprocs=N_GPUS,
             join=True)
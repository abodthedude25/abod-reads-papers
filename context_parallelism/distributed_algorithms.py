import torch
import torch.distributed as dist
from attention import simple_attention, merge_attention_outputs
import time

def run_ring_pass_kv_distributed(rank, world_size, q_shard, k_shard, v_shard, global_indices, profile=False):
    """
    A true distributed implementation of Ring Pass-KV (Algorithm 2).
    
    This version uses BLOCKING send/recv. This is simpler, easier to debug,
    and avoids deadlocks and race conditions.
    
    It also fixes the 'cudaErrorIllegalAddress' by casting the 'long'
    index tensor to 'float32' before sending.
    """
    
    O_partials = []
    LSE_partials = []
    device = q_shard.device
    
    # --- Profiling ---
    compute_times = []
    comm_times = []

    # --- Buffers ---
    # We only need one buffer for receiving
    k_recv = torch.empty_like(k_shard, device=device)
    v_recv = torch.empty_like(v_shard, device=device)
    
    indices_recv = torch.empty_like(global_indices, device=device, dtype=torch.float32)

    # --- Initial Tensors ---
    k_curr = k_shard
    v_curr = v_shard
    indices_curr = global_indices
    
    # --- Neighbors ---
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    
    for step in range(world_size):
        
        # --- 1. Compute ---
        torch.cuda.synchronize()
        t_comp_start = time.time()

        causal_mask_chunk = global_indices.unsqueeze(-1) >= indices_curr.unsqueeze(-2)
        O_partial, LSE_partial = simple_attention(
            q_shard, k_curr, v_curr, causal_mask=causal_mask_chunk
        )
        O_partials.append(O_partial)
        LSE_partials.append(LSE_partial)
        
        torch.cuda.synchronize()
        compute_times.append((time.time() - t_comp_start) * 1000) # in ms

        # --- 2. Communicate (if not the last step) ---
        if step < world_size - 1:
            torch.cuda.synchronize()
            t_comm_start = time.time()

            # Prepare tensors to send
            k_to_send = k_curr.contiguous()
            v_to_send = v_curr.contiguous()
            
            indices_to_send = indices_curr.contiguous().float()

            # Blocking, deadlock-safe send/recv
            # Even ranks send first, odd ranks receive first
            if rank % 2 == 0:
                dist.send(k_to_send, dst=next_rank)
                dist.send(v_to_send, dst=next_rank)
                dist.send(indices_to_send, dst=next_rank) # Send as FLOAT
                
                dist.recv(k_recv, src=prev_rank)
                dist.recv(v_recv, src=prev_rank)
                dist.recv(indices_recv, src=prev_rank) # Receive as FLOAT
            else:
                dist.recv(k_recv, src=prev_rank)
                dist.recv(v_recv, src=prev_rank)
                dist.recv(indices_recv, src=prev_rank) # Receive as FLOAT
                
                dist.send(k_to_send, dst=next_rank)
                dist.send(v_to_send, dst=next_rank)
                dist.send(indices_to_send, dst=next_rank) # Send as FLOAT

            torch.cuda.synchronize()
            comm_times.append((time.time() - t_comm_start) * 1000) # in ms
            
            # Update k_curr for the next loop
            k_curr = k_recv
            v_curr = v_recv
            
            indices_curr = indices_recv.long()

    # --- Print Profiling Breakdown ---
    if profile and rank == 0:
        total_compute = sum(compute_times)
        total_comm = sum(comm_times)
        # Handle division by zero if world_size is 1
        total_time = total_compute + total_comm if comm_times else total_compute
        
        print(f"\n  📊 Detailed Timing Breakdown (Rank 0):")
        print(f"     Total compute: {total_compute:6.2f}ms ({total_compute/total_time*100:5.1f}%)")
        print(f"     Total comm:    {total_comm:6.2f}ms ({total_comm/total_time*100:5.1f}%)")
        print(f"     ─────────────────────────────────")
        print(f"     Total time:    {total_time:6.2f}ms")
        
        avg_compute = total_compute / len(compute_times)
        avg_comm = total_comm / len(comm_times) if comm_times else 0
        
        if avg_comm > avg_compute:
            ratio = avg_comm / avg_compute
            print(f"     ⚠️  Communication is {ratio:.1f}x slower than compute (no overlap).")
        else:
            print(f"     ✅ Compute is the bottleneck (as expected).")
    
    # Merge all partial outputs
    final_O_shard = merge_attention_outputs(O_partials, LSE_partials)
    return final_O_shard
import torch
from attention import simple_attention, merge_attention_outputs

def run_ring_pass_kv_prefill(ranks: list['CPRank']):
    """
    Simulates Ring Pass-KV (Algorithm 2).
    - Q is stationary.
    - KV is passed around the ring.
    """
    world_size = len(ranks)
    final_outputs = [None] * world_size
    
    # Simulate the ring network buffer holding (K, V, global_indices)
    kv_ring_buffer = [(r.k_shard, r.v_shard, r.global_indices) for r in ranks]

    for rank_id, rank in enumerate(ranks):
        O_partials = []
        LSE_partials = []
        
        for step in range(world_size):
            # Rank `rank_id` computes with KV from rank `(rank_id - step) % world_size`
            kv_src_rank = (rank_id - step + world_size) % world_size
            k_chunk, v_chunk, k_global_indices = kv_ring_buffer[kv_src_rank]

            # Build the exact causal mask for this Q_i, K_j block
            q_global_indices = rank.global_indices
            causal_mask_chunk = q_global_indices.unsqueeze(-1) >= k_global_indices.unsqueeze(-2)
            
            O_partial, LSE_partial = simple_attention(
                rank.q_shard, k_chunk, v_chunk, causal_mask=causal_mask_chunk
            )
            
            O_partials.append(O_partial)
            LSE_partials.append(LSE_partial)
            
        final_outputs[rank_id] = merge_attention_outputs(O_partials, LSE_partials)

    return final_outputs


def run_ring_pass_q_prefill(ranks: list['CPRank']):
    """
    Simulates Ring Pass-Q (Algorithm 3).
    - KV is stationary.
    - Q is passed around the ring.
    - Requires an All2All to gather scattered partial outputs.
    """
    world_size = len(ranks)
    
    # Simulates the scattered partial outputs before the All2All
    # all_partial_outputs[dest_rank][src_rank] = (O_partial, LSE_partial)
    all_partial_outputs = [[None] * world_size for _ in range(world_size)]

    # Simulate the Q ring buffer holding (Q, global_indices, original_src_rank)
    q_ring_buffer = [(r.q_shard, r.global_indices, r.rank_id) for r in ranks]

    for step in range(world_size):
        # All ranks compute in parallel with their stationary KV
        for rank_id, rank in enumerate(ranks):
            # Get the Q that has arrived at this rank
            current_q, q_global_indices, current_q_src_rank = q_ring_buffer[rank_id]
            
            # Get this rank's stationary KV
            k_chunk, v_chunk = rank.k_shard, rank.v_shard
            k_global_indices = rank.global_indices
            
            causal_mask_chunk = q_global_indices.unsqueeze(-1) >= k_global_indices.unsqueeze(-2)

            O_partial, LSE_partial = simple_attention(
                current_q, k_chunk, v_chunk, causal_mask=causal_mask_chunk
            )
            
            # Store the result, indexed by *Q's original rank*
            all_partial_outputs[current_q_src_rank][rank_id] = (O_partial, LSE_partial)

        # Simulate the ring SendRecv (rotate the buffer)
        q_ring_buffer = [q_ring_buffer[-1]] + q_ring_buffer[:-1]

    # Simulate All2All and Merge
    final_outputs = [None] * world_size
    for rank_id in range(world_size):
        # Rank `rank_id` gathers all its partials
        partials_for_this_rank = all_partial_outputs[rank_id]
        
        O_partials = [p[0] for p in partials_for_this_rank]
        LSE_partials = [p[1] for p in partials_for_this_rank]
        
        final_outputs[rank_id] = merge_attention_outputs(O_partials, LSE_partials)

    return final_outputs


def run_ring_pass_q_decode(ranks: list['CPRank'], q_token, q_global_idx, full_causal_mask):
    """
    Simulates Batched Ring Pass-Q Decode (Algorithm 4).
    - Assumes q_token (T=1) has been assigned to one rank, which
      initiates the ring pass.
    - All ranks hold their *full* KV cache stationary.
    - Returns the final O token.
    """
    world_size = len(ranks)
    
    # We only care about the single Q token and its eventual output
    O_partials = []
    LSE_partials = []
    
    # Simulate the Q ring buffer
    # In a real system, only the one Q token is passed, but we
    # simulate it as a rotating buffer for simplicity.
    # The (q_token, q_global_idx) tuple is passed.
    q_ring_buffer = [(None, -1)] * world_size
    q_ring_buffer[0] = (q_token, q_global_idx) # Q starts at Rank 0

    for step in range(world_size):
        # All ranks compute in parallel
        for rank_id, rank in enumerate(ranks):
            
            current_q, current_q_global_idx = q_ring_buffer[rank_id]
            
            # Only the rank that has the token computes
            if current_q is None:
                continue
                
            # Get this rank's *entire* stationary KV cache
            k_cache, v_cache = rank.k_cache, rank.v_cache
            k_global_indices = rank.global_indices

            if k_cache.shape[1] == 0:
                # This rank has no KV cache yet, skip computation
                continue

            # Build the causal mask for this T=1 token against this rank's KV cache
            # We can't just compare indices, as the mask is pre-built
            # The q_global_idx tells us *which row* of the full mask to use
            # The k_global_indices tell us *which columns*
            causal_mask_chunk = full_causal_mask[
                :, current_q_global_idx:current_q_global_idx+1, k_global_indices
            ]
            
            O_partial, LSE_partial = simple_attention(
                current_q, k_cache, v_cache, causal_mask=causal_mask_chunk
            )
            
            O_partials.append(O_partial)
            LSE_partials.append(LSE_partial)

        # Simulate the ring SendRecv
        q_ring_buffer = [q_ring_buffer[-1]] + q_ring_buffer[:-1]
    
    # Merge the (up to) N partial outputs for the single token
    if not O_partials:
        # This can happen if T=0 (no prefill)
        return torch.zeros_like(q_token), torch.tensor(float('-inf'))

    final_O = merge_attention_outputs(O_partials, LSE_partials)
    return final_O
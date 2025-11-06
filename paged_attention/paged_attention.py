import torch
from sequence import Sequence
from kv_cache_manager import PagedKVCacheManager
from attention import simple_attention

def gather_blocks(seq: Sequence, kv_cache_manager: PagedKVCacheManager, kv_which: int):
    """
    Gathers the K (0) or V (1) blocks for a sequence from the physical cache.
    This simulates the "gather" part of the PagedAttention kernel.
    
    Returns a contiguous tensor.
    """
    physical_cache = kv_cache_manager.physical_cache_kv
    block_table = seq.get_block_table()
    block_size = kv_cache_manager.block_size
    
    # [H, T, D]
    block_shape = physical_cache.shape[2:]
    
    # Get all blocks for this sequence
    blocks_to_cat = [physical_cache[block_id, kv_which] for block_id in block_table]
    
    # Concatenate along the token dimension (dim=1)
    # [H, T_total_in_blocks, D]
    contiguous_tensor = torch.cat(blocks_to_cat, dim=1)
    
    # Trim the padding from the last block
    return contiguous_tensor[:, :seq.logical_len, :]

def run_paged_attention(
    query_tensor: torch.Tensor, # Shape: [B, H, T_q, D_H]
    seq: Sequence,
    kv_cache_manager: PagedKVCacheManager
):
    """
    Runs a full PagedAttention operation.
    1. Gathers the scattered K/V blocks into contiguous tensors.
    2. Runs the standard attention kernel on them.
    """
    
    # 1. Gather
    # k_cache shape: [H, T_kv, D_H]
    k_cache = gather_blocks(seq, kv_cache_manager, 0)
    # v_cache shape: [H, T_kv, D_V]
    v_cache = gather_blocks(seq, kv_cache_manager, 1)

    # Add a batch dimension for simple_attention
    # [B, H, T_kv, D_H]
    k_cache = k_cache.unsqueeze(0)
    v_cache = v_cache.unsqueeze(0)

    # 2. Run Attention
    # We don't need a mask because the `gather_blocks`
    # function already trimmed the cache to the logical_len.
    # The standard kernel will attend to all T_kv, which is correct.
    O, _ = simple_attention(query_tensor, k_cache, v_cache, causal_mask=None)
    
    return O
import torch

def get_load_balanced_shards(T, N):
    """
    Implements load-balanced sharding from Section 3.5.1.
    
    For a sequence of length T and N ranks, partitions T into 2N chunks.
    Rank `i` is assigned `(Chunk_i, Chunk_2N-1-i)`.
    This balances the quadratic compute load of causal attention.
    
    Returns:
     - A list of `N` lists, where each inner list contains the
       *global indices* assigned to that rank.
    """
    # Ceiling division
    chunk_size = (T + 2*N - 1) // (2*N)
    chunks = []
    for i in range(2*N):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, T)
        if start < end:
            chunks.append(list(range(start, end)))
            
    # Pad with empty lists if T < 2*N
    while len(chunks) < 2*N:
        chunks.append([])

    rank_indices = []
    for i in range(N):
        my_indices = chunks[i] + chunks[2*N - 1 - i]
        rank_indices.append(my_indices)
        
    return rank_indices
import torch
from collections import defaultdict
from sequence import Sequence
import math

class BaselineCacheManager:
    """
    Simulates the old, contiguous memory manager (like FasterTransformer).
    It suffers from massive internal fragmentation.
    """
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.total_tokens_stored = 0
        self.total_slots_allocated = 0
        self.num_requests = 0

    def alloc(self, num_tokens_to_store: int):
        """
        Allocates memory for a new request.
        It *must* allocate for the max_seq_len.
        """
        self.total_slots_allocated += self.max_seq_len
        self.total_tokens_stored += num_tokens_to_store
        self.num_requests += 1

    def report_waste(self):
        """Prints the memory waste percentage."""
        if self.total_slots_allocated == 0:
            print("  - No memory allocated.")
            return
        
        waste = (self.total_slots_allocated - self.total_tokens_stored)
        waste_percent = (waste / self.total_slots_allocated) * 100
        
        print(f"  - Total tokens stored: {self.total_tokens_stored}")
        print(f"  - Total tokens ALLOCATED: {self.total_slots_allocated}")
        print(f"  - ❌ WASTED MEMORY (Internal Fragmentation): {waste_percent:.1f}%")


class PagedKVCacheManager:
    """
    Simulates the vLLM PagedAttention memory manager.
    It manages a pool of fixed-size blocks and uses Copy-on-Write.
    """
    def __init__(self, num_blocks: int, block_size: int, 
                 num_heads: int, head_size: int, dtype, device):
        print(f"PagedKVCacheManager initialized with {num_blocks} blocks.")
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        
        # This is the actual physical memory pool
        # Shape: [num_blocks, 2 (K/V), num_heads, block_size, head_size]
        self.physical_cache_kv = torch.empty(
            (num_blocks, 2, num_heads, block_size, head_size),
            dtype=dtype,
            device=device
        )
        
        # A list of available physical block IDs
        self.free_blocks = list(range(num_blocks))
        
        # Tracks sharing. Maps: physical_block_id -> ref_count
        self.block_ref_counts = defaultdict(int)
        
        # Tracks all active sequences. Maps: seq_id -> Sequence object
        self.sequences = {}
    
    def _alloc_block(self) -> int:
        """Allocates a single free block from the pool."""
        if not self.free_blocks:
            raise MemoryError("Out of memory: No free blocks available.")
        
        block_id = self.free_blocks.pop()
        self.block_ref_counts[block_id] = 1
        return block_id

    def _free_block(self, block_id: int):
        """Frees a block, returning it to the pool if ref count is 0."""
        if block_id not in self.block_ref_counts:
            print(f"Warning: Trying to free a block ({block_id}) not in use.")
            return
            
        self.block_ref_counts[block_id] -= 1
        
        if self.block_ref_counts[block_id] == 0:
            del self.block_ref_counts[block_id]
            self.free_blocks.append(block_id)

    def alloc_sequence(self, seq_id: int, num_prompt_tokens: int) -> Sequence:
        """Allocates a new sequence for a prompt."""
        if seq_id in self.sequences:
            raise ValueError(f"Sequence {seq_id} already exists.")
            
        seq = Sequence(seq_id, self.block_size)
        seq.logical_len = num_prompt_tokens
        
        num_blocks_needed = math.ceil(num_prompt_tokens / self.block_size)
        
        # --- FIX for empty prompts ---
        # If prompt is 0 tokens, don't allocate a block yet.
        if num_prompt_tokens == 0:
             num_blocks_needed = 0
        # --- END FIX ---
            
        for _ in range(num_blocks_needed):
            block_id = self._alloc_block()
            seq.append_block(block_id)
            
        self.sequences[seq_id] = seq
        return seq

    def free_sequence(self, seq_id: int):
        """Frees all blocks associated with a sequence."""
        if seq_id not in self.sequences:
            print(f"Warning: Trying to free non-existent sequence {seq_id}.")
            return
            
        seq = self.sequences[seq_id]
        for block_id in seq.get_block_table():
            self._free_block(block_id)
        
        del self.sequences[seq_id]

    def fork(self, parent_seq_id: int, new_seq_id: int) -> Sequence:
        """
        Creates a new sequence that *shares* all blocks with the parent.
        This is the core of Copy-on-Write.
        """
        if parent_seq_id not in self.sequences:
            raise ValueError(f"Parent sequence {parent_seq_id} does not exist.")
            
        parent_seq = self.sequences[parent_seq_id]
        
        # Create the new child sequence
        child_seq = Sequence(new_seq_id, self.block_size)
        child_seq.logical_len = parent_seq.logical_len
        child_seq.block_table = list(parent_seq.get_block_table()) # Copy the block table
        
        # Increment reference counts for all shared blocks
        for block_id in child_seq.get_block_table():
            self.block_ref_counts[block_id] += 1
            
        self.sequences[new_seq_id] = child_seq
        return child_seq

    def _copy_on_write(self, seq: Sequence, block_to_copy_id: int):
        """
        Performs the Copy-on-Write operation.
        1. Allocates a new block
        2. Copies data from the old block to the new one
        3. Frees the old block (decrementing its ref count)
        4. Updates the sequence's block table to point to the new block
        """
        new_block_id = self._alloc_block()
        
        # Simulate the copy
        self.physical_cache_kv[new_block_id] = self.physical_cache_kv[block_to_copy_id].clone()
        
        # --- THIS IS THE FIX ---
        # The block to copy is ALWAYS the last block in the table.
        if seq.block_table[-1] != block_to_copy_id:
             # This should ideally not happen, but a good sanity check
             raise LogicError("CoW trying to copy a block that isn't the last block.")
        
        seq.block_table[-1] = new_block_id
        # --- END FIX ---
                
        # Decrement the old block's ref count
        self._free_block(block_to_copy_id)
        return new_block_id

    def append(self, seq_id: int):
        """
        Simulates generating one new token for a sequence.
        This handles all the logic for allocation and Copy-on-Write.
        """
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} does not exist.")
            
        seq = self.sequences[seq_id]
        
        if seq.is_last_block_full():
            # The last block is full, so we *must* allocate a new one.
            # No CoW needed here.
            new_block_id = self._alloc_block()
            seq.append_block(new_block_id)
        else:
            # The last block has space. We must check if it's shared.
            last_block_id = seq.get_last_physical_block()
            
            # --- FIX for empty sequences (len 0) ---
            if last_block_id is None:
                new_block_id = self._alloc_block()
                seq.append_block(new_block_id)
            # --- END FIX ---
            elif self.block_ref_counts[last_block_id] > 1:
                # This block is shared! We must perform Copy-on-Write.
                self._copy_on_write(seq, last_block_id)
        
        # Increment the logical length
        seq.logical_len += 1

    def report_waste(self):
        """
        Prints the memory waste percentage for the paged manager.
        NOTE: This metric is only meaningful for NON-SHARING scenarios.
        """
        total_tokens_stored = sum(seq.logical_len for seq in self.sequences.values())
        total_blocks_used = len(self.block_ref_counts)
        total_tokens_allocated = total_blocks_used * self.block_size
        
        if total_tokens_allocated == 0:
            print("  - No memory allocated.")
            return

        # Waste is only the padding in the *last* block of each sequence
        waste = (total_tokens_allocated - total_tokens_stored)
        waste_percent = (waste / total_tokens_allocated) * 100
        
        print(f"  - Total tokens stored: {total_tokens_stored}")
        print(f"  - Total blocks used: {total_blocks_used}")
        print(f"  - Total tokens ALLOCATED: {total_tokens_allocated} ({total_blocks_used} blocks * {self.block_size} tokens)")
        
        # Handle the CoW case where this metric is nonsensical
        if waste < 0:
             print(f"  - ⚠️ WASTED MEMORY (Padding): N/A (metric invalid due to CoW sharing)")
        else:
             print(f"  - ✅ WASTED MEMORY (Padding): {waste_percent:.1f}%")

    def get_total_blocks_used(self):
        return len(self.block_ref_counts)

    def print_ref_counts(self):
        """Helper for debugging CoW."""
        # Sort for consistent output
        sorted_refs = sorted(self.block_ref_counts.items())
        print(f"    - Ref Counts: {dict(sorted_refs)}")
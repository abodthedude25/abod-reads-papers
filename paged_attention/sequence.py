import torch
from typing import List

class Sequence:
    """
    Represents a single sequence (e.g., a user request).
    It stores the logical state of the sequence and its mapping
    to physical blocks via the block_table.
    """
    def __init__(self, seq_id: int, block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.logical_len = 0
        self.block_table: List[int] = [] # This is the "page table"

    def append_block(self, block_id: int):
        """Adds a new physical block to this sequence's table."""
        self.block_table.append(block_id)

    def get_last_physical_block(self) -> int:
        """Returns the ID of the last physical block."""
        if not self.block_table:
            return None
        return self.block_table[-1]

    def get_block_table(self) -> List[int]:
        """Returns the full list of physical block IDs."""
        return self.block_table
    
    def get_num_tokens_in_last_block(self) -> int:
        """Calculates how many tokens are used in the last block."""
        if self.logical_len == 0:
            return 0
        return (self.logical_len - 1) % self.block_size + 1
    
    def is_last_block_full(self) -> bool:
        """Checks if the last physical block is full."""
        if self.logical_len == 0:
            return False # An empty sequence never has a full block
        return self.get_num_tokens_in_last_block() == self.block_size

    def __repr__(self):
        return (f"Sequence(id={self.seq_id}, "
                f"logical_len={self.logical_len}, "
                f"blocks={self.block_table})")
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    """Initializes the distributed environment."""
    # These environment variables are needed for 'nccl' backend
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    # 'nccl' is the standard, high-performance backend for GPU-to-GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Pin the process to the correct GPU
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized on GPU {torch.cuda.current_device()}")

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()
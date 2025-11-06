"""
Stream Batch: Batch denoising steps diagonally across frames.

Key Insight:
- Instead of: Frame1(step1→step2→...→stepN), Frame2(step1→...→stepN)
- Do: Batch([Frame1-step1, Frame2-step1, ...]), Batch([Frame1-step2, Frame2-step2, ...])
- Result: Process multiple frames in parallel with single U-Net passes
"""

import torch
from typing import List, Tuple
from collections import deque


class StreamBatch:
    """
    Implements diagonal batch processing for continuous frame streams.
    
    Example with 3 frames, 4 denoising steps:
        Frame 1: [S0] [S1] [S2] [S3]
        Frame 2:      [S0] [S1] [S2] [S3]
        Frame 3:           [S0] [S1] [S2] [S3]
        
    Batch 0: [F1-S0]
    Batch 1: [F1-S1, F2-S0]
    Batch 2: [F1-S2, F2-S1, F3-S0]
    Batch 3: [F1-S3, F2-S2, F3-S1]
    Batch 4: [F2-S3, F3-S2]
    Batch 5: [F3-S3]
    """
    
    def __init__(self, num_denoising_steps: int, batch_size: int = None):
        """
        Args:
            num_denoising_steps: Number of denoising steps (n)
            batch_size: Maximum batch size (defaults to num_denoising_steps)
        """
        self.num_steps = num_denoising_steps
        self.batch_size = batch_size or num_denoising_steps
        
        # Queue structure: each element is (latent, current_step, frame_id)
        self.queue = deque(maxlen=self.num_steps)
        self.frame_counter = 0
        
    def add_frame(self, latent: torch.Tensor) -> None:
        """Add a new frame to the stream at step 0."""
        self.queue.append({
            'latent': latent,
            'step': 0,
            'frame_id': self.frame_counter
        })
        self.frame_counter += 1
        
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Get the current batch for U-Net processing.
        
        Returns:
            latents: Batched latent tensors [batch_size, C, H, W]
            timesteps: Timesteps for each element [batch_size]
            frame_ids: Frame IDs for tracking [batch_size]
        """
        if len(self.queue) == 0:
            return None, None, None
            
        batch_latents = []
        batch_timesteps = []
        batch_frame_ids = []
        
        for item in self.queue:
            batch_latents.append(item['latent'])
            batch_timesteps.append(item['step'])
            batch_frame_ids.append(item['frame_id'])
            
        # Stack into batched tensors
        latents = torch.stack(batch_latents, dim=0)
        timesteps = torch.tensor(batch_timesteps, device=latents.device)
        
        return latents, timesteps, batch_frame_ids
        
    def update_batch(self, denoised_latents: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """
        Update queue with denoised results and advance steps.
        
        Args:
            denoised_latents: Denoised batch from U-Net [batch_size, C, H, W]
            
        Returns:
            completed_frames: List of (latent, frame_id) for finished frames
        """
        completed = []
        
        for i, item in enumerate(self.queue):
            # Update latent with denoised result
            item['latent'] = denoised_latents[i]
            item['step'] += 1
            
            # Check if this frame is complete
            if item['step'] >= self.num_steps:
                completed.append((item['latent'], item['frame_id']))
                
        # Remove completed frames from queue
        self.queue = deque([item for item in self.queue if item['step'] < self.num_steps])
        
        return completed
        
    def is_warmed_up(self) -> bool:
        """Check if queue has reached steady state."""
        return len(self.queue) == min(self.batch_size, self.num_steps)
        
    def reset(self):
        """Clear the queue."""
        self.queue.clear()
        self.frame_counter = 0


def demonstrate_stream_batch():
    """Educational demonstration of Stream Batch concept."""
    print("=== Stream Batch Demonstration ===\n")
    
    num_steps = 4
    sb = StreamBatch(num_denoising_steps=num_steps)
    
    # Simulate 6 frame inputs
    print("Adding frames to stream:")
    for i in range(6):
        # Mock latent (just for demonstration)
        latent = torch.randn(1, 4, 64, 64)
        sb.add_frame(latent)
        print(f"  Frame {i} added (step 0)")
        
        # Process batch
        latents, timesteps, frame_ids = sb.get_batch()
        if latents is not None:
            print(f"  → Batch: frames {frame_ids} at steps {timesteps.tolist()}")
            
            # Simulate U-Net processing (just pass through for demo)
            denoised = latents  # In reality: denoised = unet(latents, timesteps)
            
            # Update queue
            completed = sb.update_batch(denoised)
            if completed:
                for latent, fid in completed:
                    print(f"  ✅ Frame {fid} completed!")
        print()


if __name__ == "__main__":
    demonstrate_stream_batch()
"""
Stochastic Similarity Filter (SSF)

Key Insight:
- Skip processing when consecutive frames are nearly identical
- Use probabilistic sampling for smooth video (avoid hard cutoffs)
- Energy savings: 2.39× on RTX 3060, 1.99× on RTX 4090

Formula (Eq. 9 in paper):
P(skip | I_t, I_ref) = max(0, (similarity - η) / (1 - η))

where similarity = cosine_similarity(I_t, I_ref)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class StochasticSimilarityFilter:
    """
    Energy-efficient frame filtering based on inter-frame similarity.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.98,
        enabled: bool = True
    ):
        """
        Args:
            similarity_threshold: η in paper (default 0.98)
            enabled: Whether SSF is active
        """
        self.eta = similarity_threshold
        self.enabled = enabled
        
        # Reference frame for comparison
        self.reference_frame = None
        self.frames_processed = 0
        self.frames_skipped = 0
        
    def compute_cosine_similarity(
        self,
        frame_current: torch.Tensor,
        frame_reference: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two frames.
        
        From Eq. 8 in paper:
        SC(I_t, I_ref) = (I_t · I_ref) / (||I_t|| ||I_ref||)
        
        Args:
            frame_current: Current input frame [C, H, W]
            frame_reference: Reference frame [C, H, W]
            
        Returns:
            Cosine similarity [0, 1]
        """
        # Flatten to vectors
        curr_flat = frame_current.flatten()
        ref_flat = frame_reference.flatten()
        
        # Cosine similarity
        similarity = F.cosine_similarity(
            curr_flat.unsqueeze(0),
            ref_flat.unsqueeze(0),
            dim=1
        ).item()
        
        return similarity
        
    def compute_skip_probability(self, similarity: float) -> float:
        """
        Compute probability of skipping based on similarity.
        
        From Eq. 9 in paper:
        P(skip | I_t, I_ref) = max(0, (similarity - η) / (1 - η))
        
        Args:
            similarity: Cosine similarity [0, 1]
            
        Returns:
            Skip probability [0, 1]
        """
        if similarity < self.eta:
            return 0.0
            
        skip_prob = (similarity - self.eta) / (1 - self.eta)
        return max(0.0, min(1.0, skip_prob))  # Clamp to [0, 1]
        
    def should_process(self, frame: torch.Tensor) -> Tuple[bool, float]:
        """
        Decide whether to process this frame or skip it.
        
        Args:
            frame: Input frame [C, H, W]
            
        Returns:
            should_process: True if frame should be processed
            similarity: Computed similarity score
        """
        if not self.enabled:
            return True, 1.0
            
        # First frame is always processed
        if self.reference_frame is None:
            self.reference_frame = frame.clone()
            self.frames_processed += 1
            return True, 0.0
            
        # Compute similarity
        similarity = self.compute_cosine_similarity(frame, self.reference_frame)
        
        # Compute skip probability
        skip_prob = self.compute_skip_probability(similarity)
        
        # Stochastic decision
        if torch.rand(1).item() < skip_prob:
            # Skip this frame
            self.frames_skipped += 1
            return False, similarity
        else:
            # Process this frame and update reference
            self.reference_frame = frame.clone()
            self.frames_processed += 1
            return True, similarity
            
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        total = self.frames_processed + self.frames_skipped
        skip_rate = self.frames_skipped / total if total > 0 else 0
        
        return {
            'processed': self.frames_processed,
            'skipped': self.frames_skipped,
            'total': total,
            'skip_rate': skip_rate,
            'energy_savings': 1 / (1 - skip_rate) if skip_rate < 1 else float('inf')
        }
        
    def reset(self):
        """Reset filter state."""
        self.reference_frame = None
        self.frames_processed = 0
        self.frames_skipped = 0


def demonstrate_ssf():
    """Educational demonstration of SSF behavior."""
    print("=== Stochastic Similarity Filter Demo ===\n")
    
    ssf = StochasticSimilarityFilter(similarity_threshold=0.98)
    
    # Simulate frame stream
    print("Simulating video stream:\n")
    
    # Static scene (high similarity)
    print("Static scene (frames 0-10):")
    static_frame = torch.randn(3, 64, 64)
    for i in range(10):
        # Add tiny noise to simulate camera noise
        noisy_frame = static_frame + torch.randn_like(static_frame) * 0.01
        should_process, similarity = ssf.should_process(noisy_frame)
        status = "✅ PROCESS" if should_process else "⏩ SKIP"
        print(f"  Frame {i}: similarity={similarity:.4f}, {status}")
    
    print()
    
    # Dynamic scene (low similarity)
    print("Dynamic scene (frames 11-15):")
    for i in range(11, 16):
        dynamic_frame = torch.randn(3, 64, 64)  # Completely different
        should_process, similarity = ssf.should_process(dynamic_frame)
        status = "✅ PROCESS" if should_process else "⏩ SKIP"
        print(f"  Frame {i}: similarity={similarity:.4f}, {status}")
    
    # Statistics
    stats = ssf.get_statistics()
    print(f"\n{'='*50}")
    print(f"Statistics:")
    print(f"  Frames processed: {stats['processed']}")
    print(f"  Frames skipped: {stats['skipped']}")
    print(f"  Skip rate: {stats['skip_rate']*100:.1f}%")
    print(f"  Energy savings: {stats['energy_savings']:.2f}×")


if __name__ == "__main__":
    demonstrate_ssf()
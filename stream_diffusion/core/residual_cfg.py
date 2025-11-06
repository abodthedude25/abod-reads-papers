"""
Residual Classifier-Free Guidance (R-CFG)

Key Insight:
- Standard CFG: noise = noise_uncond + γ × (noise_cond - noise_uncond)
  Requires 2n U-Net calls (n for cond, n for uncond)
  
- R-CFG: Approximate noise_uncond using the original input x₀
  ε_neg ≈ (x_τ - √α_τ · x₀) / √β_τ
  Requires only n U-Net calls (analytical for uncond!)
  
Two variants:
1. Self-Negative: Use original x₀ → 0 extra U-Net calls
2. Onetime-Negative: Compute once at first step → 1 extra U-Net call
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ResidualCFG:
    """
    Residual Classifier-Free Guidance for fast conditional generation.
    """
    
    def __init__(
        self, 
        guidance_scale: float = 7.5,
        delta: float = 1.0,
        method: str = "self_negative"
    ):
        """
        Args:
            guidance_scale: CFG strength (γ in paper)
            delta: Magnitude moderation coefficient for virtual residual
            method: "self_negative" or "onetime_negative"
        """
        self.gamma = guidance_scale
        self.delta = delta
        self.method = method
        
        # Cache for onetime negative
        self.cached_x0_neg = None
        self.cached_negative_computed = False
        
    def compute_virtual_negative_noise(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        alpha_t: float,
        beta_t: float
    ) -> torch.Tensor:
        """
        Analytically compute virtual negative noise pointing toward x_0.
        
        From Eq. 5 in paper:
        ε_τ,c̄' = (x_τ - √α_τ · x₀) / √β_τ
        
        Args:
            x_t: Current noisy latent
            x_0: Original input latent (or negative-conditioned reference)
            alpha_t: α_τ from noise scheduler
            beta_t: β_τ from noise scheduler
            
        Returns:
            Virtual negative noise
        """
        sqrt_alpha = torch.sqrt(torch.tensor(alpha_t, device=x_t.device))
        sqrt_beta = torch.sqrt(torch.tensor(beta_t, device=x_t.device))
        
        # Equation 5 from paper
        epsilon_neg = (x_t - sqrt_alpha * x_0) / sqrt_beta
        
        return epsilon_neg
        
    def apply_guidance(
        self,
        epsilon_cond: torch.Tensor,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        alpha_t: float,
        beta_t: float,
        epsilon_uncond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply R-CFG to get final guided noise prediction.
        
        From Eq. 6 in paper:
        ε_τ,cfg = δ·ε_τ,c̄' + γ·(ε_τ,c - δ·ε_τ,c̄')
        
        Args:
            epsilon_cond: Conditional noise from U-Net
            x_t: Current noisy latent
            x_0: Original input (self-negative) or negative reference (onetime)
            alpha_t, beta_t: Scheduler parameters
            epsilon_uncond: Optional precomputed negative (for onetime method)
            
        Returns:
            Guided noise prediction
        """
        # Compute virtual negative noise
        epsilon_neg_virtual = self.compute_virtual_negative_noise(
            x_t, x_0, alpha_t, beta_t
        )
        
        # Apply guidance formula (Eq. 6)
        epsilon_cfg = (
            self.delta * epsilon_neg_virtual + 
            self.gamma * (epsilon_cond - self.delta * epsilon_neg_virtual)
        )
        
        return epsilon_cfg
        
    def setup_onetime_negative(
        self,
        unet: nn.Module,
        x_t0: torch.Tensor,
        timestep_0: int,
        negative_embedding: torch.Tensor,
        alpha_t0: float,
        beta_t0: float
    ) -> None:
        """
        For onetime-negative method: Compute negative reference once at start.
        
        From Eq. 7 in paper:
        x̂₀,τ₀,c̄ = (x_τ₀ - √β_τ₀ · ε_τ₀,c̄) / √α_τ₀
        
        Args:
            unet: The U-Net model
            x_t0: Initial noisy latent
            timestep_0: Initial timestep
            negative_embedding: Negative conditioning embedding
            alpha_t0, beta_t0: Scheduler parameters at t=0
        """
        # Compute negative noise ONCE
        with torch.no_grad():
            epsilon_uncond = unet(x_t0, timestep_0, negative_embedding)
            
        # Compute reference point (Eq. 7)
        sqrt_alpha = torch.sqrt(torch.tensor(alpha_t0, device=x_t0.device))
        sqrt_beta = torch.sqrt(torch.tensor(beta_t0, device=x_t0.device))
        
        self.cached_x0_neg = (x_t0 - sqrt_beta * epsilon_uncond) / sqrt_alpha
        self.cached_negative_computed = True
        
    def reset(self):
        """Reset cache for new generation."""
        self.cached_x0_neg = None
        self.cached_negative_computed = False


def compare_cfg_methods():
    """
    Educational comparison of CFG computation requirements.
    """
    print("=== CFG Method Comparison ===\n")
    
    num_steps = 5
    
    print(f"Denoising steps: {num_steps}\n")
    
    # Standard CFG
    standard_unet_calls = 2 * num_steps
    print(f"Standard CFG:")
    print(f"  U-Net calls: {num_steps} (cond) + {num_steps} (uncond) = {standard_unet_calls}")
    print(f"  Computation: 2× baseline")
    
    # Self-Negative R-CFG  
    self_neg_calls = num_steps
    print(f"\nSelf-Negative R-CFG:")
    print(f"  U-Net calls: {num_steps} (cond) + 0 (analytical) = {self_neg_calls}")
    print(f"  Computation: 1× baseline")
    print(f"  Speedup: {standard_unet_calls / self_neg_calls:.2f}×")
    
    # Onetime-Negative R-CFG
    onetime_calls = num_steps + 1
    print(f"\nOnetime-Negative R-CFG:")
    print(f"  U-Net calls: {num_steps} (cond) + 1 (uncond, first step only) = {onetime_calls}")
    print(f"  Computation: {onetime_calls/num_steps:.2f}× baseline")
    print(f"  Speedup: {standard_unet_calls / onetime_calls:.2f}×")
    
    print("\n" + "="*50)
    print("Key Insight: R-CFG trades slight approximation for massive speedup!")


if __name__ == "__main__":
    compare_cfg_methods()
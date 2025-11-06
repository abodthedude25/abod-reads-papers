"""
Pre-computation Cache Manager

Key Insight:
- Many computations are static across frames (prompt embeddings, noise schedules, etc.)
- Compute once, cache, and reuse → reduce redundant computation

Cached items:
1. Prompt embeddings (text encoder output)
2. Noise schedule coefficients (α_t, β_t)
3. Pre-sampled Gaussian noise
4. Key/Value pairs in U-Net cross-attention
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


class CacheManager:
    """
    Manages pre-computed values to avoid redundant computation.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Args:
            device: Device to store cached tensors
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache storage
        self.prompt_embeddings_cache = {}
        self.noise_schedule_cache = {}
        self.presampled_noise_cache = {}
        self.attention_kv_cache = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def cache_prompt_embedding(
        self,
        prompt: str,
        embedding: torch.Tensor
    ) -> None:
        """
        Cache text encoder output for a prompt.
        
        Args:
            prompt: Text prompt
            embedding: Encoded prompt embedding [seq_len, hidden_dim]
        """
        self.prompt_embeddings_cache[prompt] = embedding.to(self.device)
        
    def get_prompt_embedding(self, prompt: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached prompt embedding.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Cached embedding or None if not found
        """
        if prompt in self.prompt_embeddings_cache:
            self.cache_hits += 1
            return self.prompt_embeddings_cache[prompt]
        else:
            self.cache_misses += 1
            return None
            
    def cache_noise_schedule(
        self,
        num_steps: int,
        alphas: torch.Tensor,
        betas: torch.Tensor
    ) -> None:
        """
        Cache noise schedule coefficients.
        
        From Eq. 10 in paper:
        x_t = √α_t · x_0 + √β_t · ε
        
        Args:
            num_steps: Number of denoising steps
            alphas: α_t values for each step
            betas: β_t values for each step
        """
        self.noise_schedule_cache[num_steps] = {
            'alphas': alphas.to(self.device),
            'betas': betas.to(self.device)
        }
        
    def get_noise_schedule(
        self,
        num_steps: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached noise schedule.
        
        Args:
            num_steps: Number of denoising steps
            
        Returns:
            (alphas, betas) or None if not found
        """
        if num_steps in self.noise_schedule_cache:
            self.cache_hits += 1
            schedule = self.noise_schedule_cache[num_steps]
            return schedule['alphas'], schedule['betas']
        else:
            self.cache_misses += 1
            return None
            
    def cache_presampled_noise(
        self,
        step: int,
        noise: torch.Tensor
    ) -> None:
        """
        Cache pre-sampled Gaussian noise for a denoising step.
        
        Key Insight: For image-to-image with consistent input,
        we can reuse the same noise across frames for stability.
        
        Args:
            step: Denoising step index
            noise: Sampled noise tensor
        """
        self.presampled_noise_cache[step] = noise.to(self.device)
        
    def get_presampled_noise(
        self,
        step: int,
        shape: Tuple[int, ...] = None
    ) -> Optional[torch.Tensor]:
        """
        Retrieve or generate cached noise.
        
        Args:
            step: Denoising step index
            shape: Shape of noise tensor (if generating new)
            
        Returns:
            Cached or newly generated noise
        """
        if step in self.presampled_noise_cache:
            self.cache_hits += 1
            return self.presampled_noise_cache[step]
        else:
            self.cache_misses += 1
            if shape is not None:
                # Generate and cache new noise
                noise = torch.randn(shape, device=self.device)
                self.cache_presampled_noise(step, noise)
                return noise
            return None
            
    def cache_attention_kv(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Cache Key/Value pairs from cross-attention layers.
        
        Key Insight: For fixed prompts, K and V in cross-attention
        remain constant → compute once and reuse.
        
        Args:
            layer_name: Identifier for the attention layer
            key: Key tensor
            value: Value tensor
        """
        self.attention_kv_cache[layer_name] = {
            'key': key.to(self.device),
            'value': value.to(self.device)
        }
        
    def get_attention_kv(
        self,
        layer_name: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached K/V pairs.
        
        Args:
            layer_name: Identifier for the attention layer
            
        Returns:
            (key, value) or None if not found
        """
        if layer_name in self.attention_kv_cache:
            self.cache_hits += 1
            kv = self.attention_kv_cache[layer_name]
            return kv['key'], kv['value']
        else:
            self.cache_misses += 1
            return None
            
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear specified cache or all caches.
        
        Args:
            cache_type: 'prompts', 'noise_schedule', 'presampled_noise', 'attention_kv', or None for all
        """
        if cache_type == 'prompts' or cache_type is None:
            self.prompt_embeddings_cache.clear()
        if cache_type == 'noise_schedule' or cache_type is None:
            self.noise_schedule_cache.clear()
        if cache_type == 'presampled_noise' or cache_type is None:
            self.presampled_noise_cache.clear()
        if cache_type == 'attention_kv' or cache_type is None:
            self.attention_kv_cache.clear()
            
        if cache_type is None:
            self.cache_hits = 0
            self.cache_misses = 0
            
    def get_statistics(self) -> dict:
        """Get cache statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'prompt_cache_size': len(self.prompt_embeddings_cache),
            'noise_schedule_cache_size': len(self.noise_schedule_cache),
            'presampled_noise_cache_size': len(self.presampled_noise_cache),
            'attention_kv_cache_size': len(self.attention_kv_cache)
        }


def demonstrate_cache_manager():
    """Educational demonstration of cache manager."""
    print("=== Cache Manager Demo ===\n")
    
    cache = CacheManager()
    
    # Simulate prompt caching
    print("1. Prompt Embedding Cache:")
    prompt = "a beautiful sunset"
    embedding = torch.randn(77, 768)  # Mock embedding
    
    cache.cache_prompt_embedding(prompt, embedding)
    print(f"   Cached: '{prompt}'")
    
    # Retrieve (cache hit)
    retrieved = cache.get_prompt_embedding(prompt)
    print(f"   Retrieved: shape={retrieved.shape} ✅")
    
    # Try different prompt (cache miss)
    different = cache.get_prompt_embedding("different prompt")
    print(f"   Different prompt: {different} ❌\n")
    
    # Simulate noise schedule caching
    print("2. Noise Schedule Cache:")
    alphas = torch.linspace(0.9, 0.1, 10)
    betas = 1 - alphas
    cache.cache_noise_schedule(10, alphas, betas)
    print(f"   Cached schedule for 10 steps")
    
    retrieved_alphas, retrieved_betas = cache.get_noise_schedule(10)
    print(f"   Retrieved: alphas shape={retrieved_alphas.shape} ✅\n")
    
    # Statistics
    stats = cache.get_statistics()
    print(f"{'='*50}")
    print(f"Cache Statistics:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"  Total cached items: {stats['prompt_cache_size'] + stats['noise_schedule_cache_size']}")


if __name__ == "__main__":
    demonstrate_cache_manager()
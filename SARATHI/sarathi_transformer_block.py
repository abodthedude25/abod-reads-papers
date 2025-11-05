import torch
import torch.nn as nn
import math

"""
This file defines the *actual* LLM code (a single, functional Transformer Block)
that is capable of running SARATHI's fused batching.
"""

class MinimalAttention(nn.Module):
    """A minimal, functional Multi-Head Attention module with KV Caching."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, causal=True):
        """
        x shape: [b, s, d_model]
        past_kv: (k, v) tuple of shapes [b, s_past, d_model]
        """
        b, s, d = x.shape
        
        # 1. Project to Q, K, V
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        # 2. Reshape for Multi-Head
        # [b, s, d_model] -> [b, s, n_heads, d_head] -> [b, n_heads, s, d_head]
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        
        # 3. Handle KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            # [b, n_heads, s_past, d_head]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # The new, complete KV cache for this step
        current_kv = (k, v)
        
        # 4. Scaled Dot-Product Attention
        s_past = k.shape[2]
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        
        # 5. Apply Causal Mask (if needed)
        if causal:
            # Mask shape [s, s_past]
            mask = torch.triu(torch.ones(s, s_past, device=x.device), diagonal=1 + s_past - s).bool()
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
            
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # 6. Get output
        # [b, n_heads, s, d_head]
        attn_output = attn_weights @ v
        
        # 7. Reshape and project output
        # [b, n_heads, s, d_head] -> [b, s, n_heads, d_head] -> [b, s, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s, d)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, current_kv

class MinimalFFN(nn.Module):
    """A minimal, functional FFN (MLP) block."""
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

class SarathiTransformerBlock(nn.Module):
    """
    A single Transformer Block that implements the SARATHI logic.
    It has three `forward` methods:
    1. forward_prefill: Baseline full prefill.
    2. forward_decode: Baseline batched decode.
    3. forward_sarathi_fused: The "piggybacking" fused batch.
    """
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.attn = MinimalAttention(d_model, n_heads)
        self.ffn = MinimalFFN(d_model, d_ffn)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward_prefill(self, tokens, past_kv=None):
        """
        Standard baseline prefill (or chunked prefill).
        tokens: [b, s, d_model]
        past_kv: Optional (k, v) tuple
        """
        # Attention block
        resid_pre_attn = self.ln1(tokens)
        attn_out, new_kv = self.attn(resid_pre_attn, past_kv=past_kv, causal=True)
        tokens = tokens + attn_out
        
        # FFN block
        resid_pre_ffn = self.ln2(tokens)
        ffn_out = self.ffn(resid_pre_ffn)
        tokens = tokens + ffn_out
        
        return tokens, new_kv

    def forward_decode(self, tokens, past_kv):
        """
        Standard baseline decode.
        tokens: [b, 1, d_model] (one token per request in batch)
        past_kv: (k, v) tuple
        """
        # Attention block
        resid_pre_attn = self.ln1(tokens)
        attn_out, new_kv = self.attn(resid_pre_attn, past_kv=past_kv, causal=False)
        tokens = tokens + attn_out
        
        # FFN block
        resid_pre_ffn = self.ln2(tokens)
        ffn_out = self.ffn(resid_pre_ffn)
        tokens = tokens + ffn_out
        
        return tokens, new_kv
        
    def forward_sarathi_fused(self, prefill_chunk, prefill_kv, decode_tokens, decode_kvs):
        """
        SARATHI's fused forward pass.
        prefill_chunk: [1, s_chunk, d_model]
        prefill_kv: (k, v) for the prefill request
        decode_tokens: [b_decode, 1, d_model]
        decode_kvs: (k_batch, v_batch) for the decode requests
        """
        
        # --- 1. Separate Attention ---
        # As the paper states, attention ops are run separately.
        
        # Process prefill chunk attention
        prefill_resid_in = self.ln1(prefill_chunk)
        prefill_attn_out, new_prefill_kv = self.attn(
            prefill_resid_in, 
            past_kv=prefill_kv, 
            causal=True
        )
        # Add residual for prefill
        prefill_ffn_in = prefill_chunk + prefill_attn_out
        
        # Process decode batch attention
        decode_resid_in = self.ln1(decode_tokens)
        decode_attn_out, new_decode_kvs = self.attn(
            decode_resid_in, 
            past_kv=decode_kvs,
            causal=False
        )
        # Add residual for decodes
        decode_ffn_in = decode_tokens + decode_attn_out

        # --- 2. Fused FFN (The "Piggyback") ---
        
        # Get shapes
        b_prefill, s_chunk, d = prefill_ffn_in.shape
        b_decode, s_decode, _ = decode_ffn_in.shape
        
        # Reshape to 2D tensors: [total_tokens, d_model]
        prefill_2d = prefill_ffn_in.view(-1, d)
        decode_2d = decode_ffn_in.view(-1, d)
        
        # Concat into one giant batch
        fused_ffn_input = torch.cat([prefill_2d, decode_2d], dim=0)
        
        # Run ONE LayerNorm and ONE FFN pass (compute-bound)
        fused_ffn_input_norm = self.ln2(fused_ffn_input)
        fused_ffn_output = self.ffn(fused_ffn_input_norm)
        
        # --- 3. Un-fuse and Add Final Residuals ---
        
        # Split the fused output back up
        prefill_ffn_out = fused_ffn_output[:s_chunk]
        decode_ffn_out = fused_ffn_output[s_chunk:]
        
        # Add final residual and reshape
        final_prefill_out = (prefill_ffn_in.view(-1, d) + prefill_ffn_out).view(b_prefill, s_chunk, d)
        final_decode_out = (decode_ffn_in.view(-1, d) + decode_ffn_out).view(b_decode, s_decode, d)
        
        return final_prefill_out, new_prefill_kv, final_decode_out, new_decode_kvs

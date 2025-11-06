import torch
import math

def simple_attention(q, k, v, causal_mask=None):
    """
    Computes standard scaled dot-product attention.
    Returns O and LSE (log-sum-exp) for numerically-stable merging.

    Shapes:
     - q: (B, H, T_q, D_H)
     - k: (B, H, T_kv, D_H)
     - v: (B, H, T_kv, D_V)
     - causal_mask: (B, T_q, T_kv) or (T_q, T_kv)
    """
    B, H, T_q, D_H = q.shape
    T_kv = k.shape[2]
    
    # Reshape for matmul: (B*H, T_q, D_H)
    q = q.reshape(B * H, T_q, D_H)
    # Reshape for matmul: (B*H, D_H, T_kv)
    k = k.transpose(-2, -1).reshape(B * H, D_H, T_kv)
    
    scores = torch.bmm(q, k) / math.sqrt(D_H) # (B*H, T_q, T_kv)
    
    if causal_mask is not None:
        mask = causal_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0) # (1, T_q, T_kv)
        # We don't need to expand for B*H, bmm will broadcast
        scores = scores.masked_fill(~mask, float('-inf'))

    # LSE (log-sum-exp) is shape (B*H, T_q)
    LSE = torch.logsumexp(scores, dim=-1)
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Handle 0/0 = nan cases (where a row is all -inf)
    attn_weights = attn_weights.nan_to_num(nan=0.0)
    
    # Reshape v for matmul: (B*H, T_kv, D_V)
    D_V = v.shape[-1]
    v = v.reshape(B * H, T_kv, D_V)
    
    O = torch.bmm(attn_weights, v) # (B*H, T_q, D_V)
    
    # Reshape O back to (B, H, T_q, D_V)
    O = O.reshape(B, H, T_q, D_V)
    
    # Reshape LSE back to (B, H, T_q)
    LSE = LSE.reshape(B, H, T_q)
    
    return O, LSE
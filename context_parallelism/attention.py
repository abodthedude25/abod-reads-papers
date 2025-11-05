import torch
import math

def simple_attention(q, k, v, causal_mask=None):
    """
    Computes standard scaled dot-product attention.
    Returns O and LSE (log-sum-exp) for numerically-stable merging.

    Shapes:
     - q: (B, T_q, D_H)
     - k: (B, T_kv, D_H)
     - v: (B, T_kv, D_V)
     - causal_mask: (B, T_q, T_kv) or (T_q, T_kv)
    """
    B, T_q, D_H = q.shape
    T_kv = k.shape[1]
    D_V = v.shape[2]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_H)
    
    if causal_mask is not None:
        # Ensure mask is broadcastable
        mask = causal_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0) # (1, T_q, T_kv)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1)
        
        # Handle cases where T_q or T_kv might not match mask, e.g., decode
        if mask.shape[1] != T_q or mask.shape[2] != T_kv:
             mask = mask[:, :T_q, :T_kv]
             
        scores = scores.masked_fill(~mask, float('-inf'))

    # LSE (log-sum-exp) is shape (B, T_q)
    LSE = torch.logsumexp(scores, dim=-1)
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    # --- THIS IS THE FIX ---
    # Handle 0/0 = nan cases (where a row is all -inf) by replacing nan with 0.0
    attn_weights = attn_weights.nan_to_num(nan=0.0)
    # --- END FIX ---
    
    O = torch.matmul(attn_weights, v)
    
    return O, LSE


def merge_attention_outputs(O_partials, LSE_partials):
    """
    Implements numerically-stable attention output merging per Appendix B, Eq. 4.

    - O_partials: List of [ (B, T_q, D_V) ] tensors
    - LSE_partials: List of [ (B, T_q) ] tensors
    """
    # LSEs shape: (N_partials, B, T_q)
    LSEs = torch.stack(LSE_partials, dim=0)
    # Os shape: (N_partials, B, T_q, D_V)
    Os = torch.stack(O_partials, dim=0)
    
    # LSE_max shape: (1, B, T_q)
    LSE_max = torch.max(LSEs, dim=0, keepdim=True)[0]
    
    # weights = exp(LSE_i - LSE_max)
    # weights shape: (N_partials, B, T_q)
    weights = torch.exp(LSEs - LSE_max)

    # O_numerator = sum(O_i * weights_i)
    # O_numerator shape: (B, T_q, D_V)
    O_numerator = torch.sum(Os * weights.unsqueeze(-1), dim=0)
    
    # O_denominator = sum(weights_i)
    # O_denominator shape: (1, B, T_q, 1) -> broadcastable
    O_denominator = torch.sum(weights, dim=0, keepdim=True).unsqueeze(-1)
    
    # Add a small epsilon to denominator to prevent 0/0 if all weights are 0
    # (e.g., if a token truly has nothing to attend to, which shouldn't happen
    # in causal, but good for robustness)
    return O_numerator / (O_denominator + 1e-9)
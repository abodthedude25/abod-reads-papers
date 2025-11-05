import torch

def heuristic_select_mode(T, P, N, C_peak, BW_peak, NKV, NH, e=2):
    """
    Implements the analytic heuristic from Algorithm 5.
    Selects pass-KV or pass-Q based on system parameters and
    the KV cache miss rate.

    - T: New tokens (T_q)
    - P: Cached KV tokens
    - N: Number of CP ranks
    - C_peak: Peak Compute (e.g., TFLOPs)
    - BW_peak: Peak Bandwidth (e.g., GB/s)
    - e: bytes per element (e.g., 2 for FP16)
    """
    
    # Handle P=0 (full prefill) case
    if P == 0:
        miss_rate = 1.0
    else:
        miss_rate = T / (T + P)
        
    # --- Directly implement Algorithm 5 logic ---
    
    try:
        # Condition 1: Is T large enough to hide KV comm latency?
        # (From Equation 2)
        cond_T_compute_bound = N * (C_peak * NKV * e) / (2 * NH * BW_peak)
        
        # Condition 2: Is the miss rate high enough?
        # (From Equation 5)
        comm_overlap_term = (4 * T * BW_peak) / (N * C_peak * e)
        cond_miss_rate_bound = 2 * (NKV / NH) - comm_overlap_term
        
    except ZeroDivisionError:
        # Fallback in case of zero division (e.g., BW_peak=0)
        cond_T_compute_bound = float('inf')
        cond_miss_rate_bound = 2 * (NKV / NH)

    # The paper's heuristic (Algorithm 5):
    # if (T >= ...) OR (miss_rate >= ...):
    #    pass-KV
    # else:
    #    pass-Q
    
    if (T >= cond_T_compute_bound) or (miss_rate >= cond_miss_rate_bound):
        return "pass-KV"
    else:
        return "pass-Q"
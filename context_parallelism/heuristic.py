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

    # 1. Message size check (from Eq 1)
    # Is pass-KV cheaper just based on tensor size?
    cond1_threshold = 2 * (NKV / NH)
    if miss_rate >= cond1_threshold:
        # e.g., T=1000, P=1000. miss_rate=0.5
        # If NKV/NH = 1/8, threshold=0.25. 0.5 > 0.25 -> pass-KV
        return "pass-KV"

    # 2. Compute-bound check (from Eq 2)
    # Is T large enough that Attn(T, T+P) compute
    # can hide the KV communication latency?
    try:
        cond2_threshold = N * (C_peak * NKV * e) / (2 * NH * BW_peak)
    except ZeroDivisionError:
        cond2_threshold = float('inf')

    if T >= cond2_threshold:
         return "pass-KV"

    # 3. Refined fallback check (from Eq 5)
    # Compares exposed pass-KV comm vs. pass-Q comm (incl. All2All)
    try:
        # This term estimates the compute time relative to bandwidth
        comm_overlap_term = (4 * T * BW_peak) / (N * C_peak * e)
        cond3_threshold = cond1_threshold - comm_overlap_term
    except ZeroDivisionError:
        cond3_threshold = cond1_threshold

    if miss_rate < cond3_threshold:
        return "pass-Q"
    else:
        # In the ambiguous zone, pass-KV is the safer default
        return "pass-KV"
import torch
import time
from sarathi_transformer_block import SarathiTransformerBlock

# --- 1. Setup Simulation Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 5120   # LLaMA-13B hidden dim
N_HEADS = 40     # LLaMA-13B num heads
D_FFN = 13824    # LLaMA-13B FFN dim
DTYPE = torch.bfloat16

# SARATHI parameters
CHUNK_SIZE = 256
DECODE_BATCH_SIZE = 17 # (for a total max batch of 18)

if DEVICE.type == 'cpu':
    print("="*60)
    print("WARNING: Running on CPU. Performance demo will not be accurate.")
    print("Please run on a CUDA-enabled GPU to see the piggybacking effect.")
    print("="*60)
    # Using smaller model for CPU to run in reasonable time
    D_MODEL, N_HEADS, D_FFN = 512, 8, 2048
    DTYPE = torch.float32

# Helper to time GPU operations correctly
def time_op(func, *args, **kwargs):
    # Warmup
    for _ in range(5):
        _ = func(*args, **kwargs)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(50): # Run 50 times for a stable average
        _ = func(*args, **kwargs)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    return (end_time - start_time) / 50 # Return average time

# --- 2. Initialize Model and Inputs ---

print("Initializing model and inputs...")
model = SarathiTransformerBlock(D_MODEL, N_HEADS, D_FFN).to(DEVICE).to(DTYPE)

# --- Test 1: Correctness Check ---
print("\n--- Test 1: Chunked Prefill Correctness Check ---")

# Full 512-token prompt
full_prompt = torch.randn(1, CHUNK_SIZE * 2, D_MODEL, device=DEVICE, dtype=DTYPE)
chunk1 = full_prompt[:, :CHUNK_SIZE, :]
chunk2 = full_prompt[:, CHUNK_SIZE:, :]

# Run as one full prefill
print("Running 1x 512-token prefill (Baseline)...")
with torch.no_grad():
    full_output, _ = model.forward_prefill(full_prompt, past_kv=None)

# Run as two chunked prefills
print("Running 2x 256-token prefill (Chunked)...")
with torch.no_grad():
    chunk1_out, kv_chunk1 = model.forward_prefill(chunk1, past_kv=None)
    chunk2_out, _ = model.forward_prefill(chunk2, past_kv=kv_chunk1)

# Combine chunked outputs
chunked_output = torch.cat([chunk1_out, chunk2_out], dim=1)

# Check for correctness
if torch.allclose(full_output, chunked_output, atol=1e-3, rtol=1e-3):
    print("\n[SUCCESS] Chunked prefill output matches full prefill output!")
else:
    print(f"\n[FAILURE] Outputs do not match! Max diff: {torch.abs(full_output - chunked_output).max()}")

# --- Test 2: Performance Demo ---
print("\n" + "--- Test 2: SARATHI Performance Demo (Piggybacking) ---")

# 1. Create Baseline Inputs
prefill_chunk1 = torch.randn(1, CHUNK_SIZE, D_MODEL, device=DEVICE, dtype=DTYPE)
prefill_chunk2 = torch.randn(1, CHUNK_SIZE, D_MODEL, device=DEVICE, dtype=DTYPE)
decode_batch_in = torch.randn(DECODE_BATCH_SIZE, 1, D_MODEL, device=DEVICE, dtype=DTYPE)

# 2. Create dummy KV Caches
# A prefill request that has already processed chunk 1
prefill_kv_cache = (
    torch.randn(1, N_HEADS, CHUNK_SIZE, D_MODEL // N_HEADS, device=DEVICE, dtype=DTYPE),
    torch.randn(1, N_HEADS, CHUNK_SIZE, D_MODEL // N_HEADS, device=DEVICE, dtype=DTYPE)
)
# A batch of decode requests, each with 256 tokens in their KV cache
decode_kv_caches = (
    torch.randn(DECODE_BATCH_SIZE, N_HEADS, 256, D_MODEL // N_HEADS, device=DEVICE, dtype=DTYPE),
    torch.randn(DECODE_BATCH_SIZE, N_HEADS, 256, D_MODEL // N_HEADS, device=DEVICE, dtype=DTYPE)
)

# 3. Run the timings
with torch.no_grad():
    # S1: Time a baseline prefill chunk
    t_prefill = time_op(model.forward_prefill, prefill_chunk1, past_kv=None)
    
    # S2: Time a baseline decode batch
    t_decode = time_op(model.forward_decode, decode_batch_in, past_kv=decode_kv_caches)
    
    # S3: Time the SARATHI fused batch
    t_sarathi = time_op(
        model.forward_sarathi_fused, 
        prefill_chunk=prefill_chunk2, 
        prefill_kv=prefill_kv_cache,
        decode_tokens=decode_batch_in,
        decode_kvs=decode_kv_caches
    )

print(f"[Baseline] Time for 1x Prefill Chunk (S1):  {t_prefill * 1000:.4f} ms")
print(f"[Baseline] Time for 1x Decode Batch (S2):   {t_decode * 1000:.4f} ms")
print(f"[SARATHI]  Time for 1x Fused Batch (S3):   {t_sarathi * 1000:.4f} ms")

print("\n--- Analysis ---")
baseline_total_time = t_prefill + t_decode
print(f"Baseline Total (S1 + S2):    {baseline_total_time * 1000:.4f} ms")
print(f"SARATHI Total (S3):          {t_sarathi * 1000:.4f} ms")
print(f"Total Speedup (Baseline / SARATHI): {baseline_total_time / t_sarathi:.2f}x")

print("\n--- The 'Aha!' Moment (Decode Cost) ---")
marginal_decode_cost = t_sarathi - t_prefill
print(f"Full cost of decodes (S2):             {t_decode * 1000:.4f} ms")
print(f"Marginal cost of decodes (S3 - S1):  {marginal_decode_cost * 1000:.4f} ms")

if marginal_decode_cost > 0 and t_decode > 0:
    speedup = t_decode / marginal_decode_cost
    print(f"\n==> Decode Piggybacking Speedup: {speedup:.2f}x <==")
else:
    print("\nCould not calculate decode speedup (timings were too small or negative).")
    print("This almost certainly means you are running on a CPU.")

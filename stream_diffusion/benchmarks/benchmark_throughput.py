"""
End-to-End Throughput Benchmarks

Compare StreamDiffusion against baseline (Diffusers AutoPipeline)
at different denoising steps, matching Table 1 from the paper.
"""

import torch
import time
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.stream_pipeline import StreamDiffusionPipeline


def benchmark_baseline(model_id: str, num_steps: int, num_frames: int = 100):
    """
    Benchmark baseline Diffusers AutoPipeline.
    
    Args:
        model_id: HuggingFace model ID
        num_steps: Number of denoising steps
        num_frames: Number of frames to benchmark
        
    Returns:
        Average time per frame in ms
    """
    print(f"Benchmarking Baseline (AutoPipeline) - {num_steps} steps...")
    
    # Load pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Create dummy input image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    
    prompt = "a photo of a cat"
    
    # Warmup
    for _ in range(3):
        _ = pipe(prompt=prompt, image=dummy_image, num_inference_steps=num_steps)
        
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    
    for _ in range(num_frames):
        _ = pipe(prompt=prompt, image=dummy_image, num_inference_steps=num_steps)
        
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / num_frames) * 1000
    
    print(f"  ✅ Baseline: {avg_time_ms:.2f} ms/frame\n")
    
    return avg_time_ms


def benchmark_streamdiffusion(model_id: str, num_steps: int, num_frames: int = 100):
    """
    Benchmark StreamDiffusion pipeline.
    
    Args:
        model_id: HuggingFace model ID
        num_steps: Number of denoising steps
        num_frames: Number of frames to benchmark
        
    Returns:
        Average time per frame in ms
    """
    print(f"Benchmarking StreamDiffusion - {num_steps} steps...")
    
    # Initialize pipeline
    pipe = StreamDiffusionPipeline(
        model_id=model_id,
        denoising_steps=num_steps,
        use_stream_batch=True,
        use_rcfg=True,
        use_ssf=False  # Not used for throughput benchmark
    )
    
    prompt = "a photo of a cat"
    
    # Warmup
    for _ in range(3):
        _ = pipe(prompt)
        
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    
    for _ in range(num_frames):
        _ = pipe(prompt)
        
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / num_frames) * 1000
    
    print(f"  ✅ StreamDiffusion: {avg_time_ms:.2f} ms/frame\n")
    
    return avg_time_ms


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmark across multiple configurations.
    Reproduces Table 1 from the paper.
    """
    print("="*70)
    print("Comprehensive Throughput Benchmark")
    print("Reproducing Table 1 from paper")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Please run on GPU for accurate benchmarks.\n")
        return
        
    model_id = "stabilityai/sd-turbo"
    steps_list = [1, 2, 4, 10]
    num_frames = 100
    
    results = []
    
    for num_steps in steps_list:
        print(f"\n{'='*70}")
        print(f"Configuration: {num_steps} denoising steps")
        print(f"{'='*70}\n")
        
        # Benchmark baseline
        baseline_time = benchmark_baseline(model_id, num_steps, num_frames)
        
        # Benchmark StreamDiffusion
        stream_time = benchmark_streamdiffusion(model_id, num_steps, num_frames)
        
        # Calculate speedup
        speedup = baseline_time / stream_time
        
        results.append({
            'steps': num_steps,
            'baseline_ms': baseline_time,
            'stream_ms': stream_time,
            'speedup': speedup
        })
        
        print(f"Speedup: {speedup:.2f}×\n")
        
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (Table 1 from paper)")
    print("="*70)
    print(f"{'Steps':<8} {'Baseline (ms)':<18} {'StreamDiff (ms)':<18} {'Speedup':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['steps']:<8} {r['baseline_ms']:<18.2f} {r['stream_ms']:<18.2f} {r['speedup']:<10.2f}×")
        
    print("="*70 + "\n")
    
    # Expected results from paper
    print("📊 Expected Results (from paper, with TensorRT):")
    print("-"*70)
    print(f"{'Steps':<8} {'AutoPipeline':<18} {'StreamDiff':<18} {'Speedup':<10}")
    print("-"*70)
    print(f"{'1':<8} {'634.40':<18} {'10.65':<18} {'59.6×':<10}")
    print(f"{'2':<8} {'652.66':<18} {'16.74':<18} {'39.3×':<10}")
    print(f"{'4':<8} {'695.20':<18} {'26.93':<18} {'25.8×':<10}")
    print(f"{'10':<8} {'803.23':<18} {'62.00':<18} {'13.0×':<10}")
    print("="*70 + "\n")
    
    print("Note: Results may differ without TensorRT optimization.")
    print("      Paper uses RTX 4090, your results depend on your GPU.\n")


def quick_benchmark():
    """Quick benchmark for testing."""
    print("="*70)
    print("Quick Throughput Test")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping benchmark\n")
        return
        
    # Quick test with SD-Turbo (1 step)
    pipe = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        denoising_steps=1,
        use_stream_batch=True,
        use_rcfg=False,
        use_ssf=False
    )
    
    results = pipe.benchmark_throughput(num_frames=50)
    
    print(f"\n🎯 Quick Test Results:")
    print(f"   FPS: {results['fps']:.2f}")
    print(f"   ms/frame: {results['ms_per_frame']:.2f}")
    
    if results['fps'] > 60:
        print(f"   ✅ Real-time capable! (>60 FPS)")
    else:
        print(f"   ⚠️  Below real-time threshold")
        
    print()


def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark StreamDiffusion throughput")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "comprehensive"],
        default="quick",
        help="Benchmark mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_benchmark()
    else:
        run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
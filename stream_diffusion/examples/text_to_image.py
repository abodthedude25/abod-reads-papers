"""
Text-to-Image Generation Example

Demonstrates basic usage of StreamDiffusion for text-to-image generation.
"""

import torch
from PIL import Image
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.stream_pipeline import StreamDiffusionPipeline


def basic_generation():
    """Basic text-to-image generation."""
    print("="*70)
    print("Text-to-Image Generation Example")
    print("="*70 + "\n")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipe = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        denoising_steps=1,  # SD-Turbo optimized for 1 step
        guidance_scale=0.0,  # SD-Turbo doesn't use CFG
        use_stream_batch=True,
        use_rcfg=False,
        use_ssf=False
    )
    
    # Generate images
    prompts = [
        "a beautiful sunset over the ocean",
        "a cute cat wearing sunglasses",
        "a futuristic cityscape at night",
        "a serene mountain landscape"
    ]
    
    print("\nGenerating images...\n")
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: '{prompt}'")
        
        # Generate
        image_tensor = pipe(prompt)
        
        # Convert to PIL and save
        image_pil = pipe.io_queue.postprocess_fn(image_tensor)
        output_path = f"outputs/output_text2img_{i+1}.png"
        image_pil.save(output_path)
        
        print(f"  ✅ Saved: {output_path}\n")
        
    print("="*70)
    print("✅ All images generated successfully!")
    print(f"📁 Check the 'outputs/' directory for results")
    print("="*70 + "\n")


def batch_generation():
    """Batch generation for maximum throughput."""
    print("="*70)
    print("Batch Generation Example (Maximum Throughput)")
    print("="*70 + "\n")
    
    pipe = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        denoising_steps=1,
        use_stream_batch=True
    )
    
    prompt = "a photo of a cat"
    num_images = 100
    
    print(f"Generating {num_images} images with prompt: '{prompt}'")
    print("This demonstrates maximum throughput...\n")
    
    import time
    start = time.time()
    
    for i in range(num_images):
        _ = pipe(prompt)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{num_images} - {fps:.2f} FPS")
            
    total_time = time.time() - start
    final_fps = num_images / total_time
    
    print(f"\n{'='*70}")
    print(f"Batch Generation Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {final_fps:.2f}")
    print(f"  ms/image: {(total_time/num_images)*1000:.2f}")
    print("="*70 + "\n")


def compare_denoising_steps():
    """Compare quality/speed trade-off with different steps."""
    print("="*70)
    print("Denoising Steps Comparison")
    print("="*70 + "\n")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    prompt = "a beautiful mountain landscape"
    steps_list = [1, 2, 4, 8]
    
    print(f"Prompt: '{prompt}'")
    print("Comparing different denoising steps...\n")
    
    for steps in steps_list:
        print(f"Steps: {steps}")
        
        pipe = StreamDiffusionPipeline(
            model_id="stabilityai/sd-turbo",
            denoising_steps=steps
        )
        
        # Time generation
        import time
        start = time.time()
        image_tensor = pipe(prompt)
        elapsed = (time.time() - start) * 1000
        
        # Save
        image_pil = pipe.io_queue.postprocess_fn(image_tensor)
        output_path = f"outputs/output_steps_{steps}.png"
        image_pil.save(output_path)
        
        print(f"  Time: {elapsed:.2f}ms")
        print(f"  Saved: {output_path}\n")
        
    print("="*70)
    print("Compare the images to see quality vs speed trade-off!")
    print(f"📁 Check the 'outputs/' directory for results")
    print("="*70 + "\n")


def main():
    """Main example entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-Image examples")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "batch", "compare"],
        default="basic",
        help="Example mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        basic_generation()
    elif args.mode == "batch":
        batch_generation()
    elif args.mode == "compare":
        compare_denoising_steps()


if __name__ == "__main__":
    main()
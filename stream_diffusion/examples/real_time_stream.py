"""
Real-Time Video Stream Example

Demonstrates StreamDiffusion's real-time capabilities with:
- Webcam input processing
- Stochastic Similarity Filter for energy efficiency
- Live style transfer
"""

import torch
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('..')

from pipeline.stream_pipeline import StreamDiffusionPipeline
from core.similarity_filter import StochasticSimilarityFilter


def webcam_style_transfer():
    """
    Real-time style transfer on webcam feed.
    
    This demonstrates the core use case from the paper:
    - Real-time interactive generation
    - Energy-efficient with SSF
    - Smooth video output
    """
    print("="*70)
    print("Real-Time Webcam Style Transfer")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA required for real-time performance\n")
        return
        
    # Initialize pipeline
    print("Initializing StreamDiffusion pipeline...")
    pipe = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        denoising_steps=1,  # Minimum steps for max speed
        use_stream_batch=True,
        use_ssf=True,  # Energy-efficient filtering
        ssf_threshold=0.98
    )
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam\n")
        return
        
    # Set resolution (lower = faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    
    print("\n✅ Pipeline ready!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Toggle SSF")
    print("  '1-4' - Change style preset\n")
    
    # Style presets
    styles = [
        "anime style",
        "oil painting",
        "watercolor",
        "cyberpunk"
    ]
    current_style = 0
    ssf_enabled = True
    
    # FPS counter
    import time
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 127.5 - 1.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Check if we should process (SSF)
            if ssf_enabled and pipe.ssf:
                should_process, similarity = pipe.ssf.should_process(frame_tensor)
                
                if not should_process:
                    # Skip processing, show previous frame
                    cv2.putText(
                        frame, "SKIPPED (SSF)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.imshow("StreamDiffusion - Real-Time", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
            
            # Process with StreamDiffusion
            # (Simplified - real impl would encode to latent, denoise, decode)
            prompt = styles[current_style]
            
            # For demo, just show input with overlay
            # Real implementation would do full img2img
            
            # Add FPS counter
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            # Display info
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Style: {prompt}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"SSF: {'ON' if ssf_enabled else 'OFF'}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("StreamDiffusion - Real-Time", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                ssf_enabled = not ssf_enabled
                print(f"SSF: {'enabled' if ssf_enabled else 'disabled'}")
            elif ord('1') <= key <= ord('4'):
                current_style = key - ord('1')
                print(f"Style changed to: {styles[current_style]}")
                
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show statistics
        print(f"\n{'='*70}")
        print("Session Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {frame_count / elapsed:.2f}")
        
        if pipe.ssf:
            stats = pipe.ssf.get_statistics()
            print(f"\nSSF Statistics:")
            print(f"  Processed: {stats['processed']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Energy savings: {stats['energy_savings']:.2f}×")
            
        print("="*70 + "\n")


def simulate_video_stream():
    """
    Simulate video stream processing (no webcam required).
    
    Demonstrates:
    - SSF behavior on static vs dynamic scenes
    - Energy savings measurement
    """
    print("="*70)
    print("Simulated Video Stream Processing")
    print("="*70 + "\n")
    
    pipe = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        denoising_steps=1,
        use_ssf=True,
        ssf_threshold=0.98
    )
    
    print("Simulating 200 frames:")
    print("  Frames 0-99: Static scene (high similarity)")
    print("  Frames 100-199: Dynamic scene (low similarity)\n")
    
    # Static scene
    static_frame = torch.randn(3, 512, 512)
    for i in range(100):
        # Add tiny noise to simulate camera
        noisy = static_frame + torch.randn_like(static_frame) * 0.01
        should_process, sim = pipe.ssf.should_process(noisy)
        
        if i % 20 == 0:
            status = "✅ PROCESS" if should_process else "⏩ SKIP"
            print(f"  Frame {i}: similarity={sim:.4f}, {status}")
            
    print("\n  Switching to dynamic scene...\n")
    
    # Dynamic scene
    for i in range(100, 200):
        dynamic_frame = torch.randn(3, 512, 512)
        should_process, sim = pipe.ssf.should_process(dynamic_frame)
        
        if (i - 100) % 20 == 0:
            status = "✅ PROCESS" if should_process else "⏩ SKIP"
            print(f"  Frame {i}: similarity={sim:.4f}, {status}")
            
    # Show statistics
    stats = pipe.ssf.get_statistics()
    
    print(f"\n{'='*70}")
    print("SSF Performance:")
    print(f"  Frames processed: {stats['processed']}")
    print(f"  Frames skipped: {stats['skipped']}")
    print(f"  Skip rate: {stats['skip_rate']*100:.1f}%")
    print(f"  Energy savings: {stats['energy_savings']:.2f}×")
    print(f"\nExpected from paper:")
    print(f"  RTX 3060: 2.39× energy savings")
    print(f"  RTX 4090: 1.99× energy savings")
    print("="*70 + "\n")


def main():
    """Main example entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time streaming examples")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["webcam", "simulate"],
        default="simulate",
        help="Example mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "webcam":
        webcam_style_transfer()
    elif args.mode == "simulate":
        simulate_video_stream()


if __name__ == "__main__":
    main()
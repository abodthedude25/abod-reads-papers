"""
Input/Output Queue System

Key Insight:
- Separate pre/post-processing from neural network inference
- Run image processing in parallel with GPU computation
- Use queues to buffer inputs and outputs

Architecture:
[Input Thread] → [Input Queue] → [GPU Pipeline] → [Output Queue] → [Output Thread]
"""

import torch
import threading
import queue
from typing import Optional, Callable, Any
import time
import numpy as np
from PIL import Image


class IOQueue:
    """
    Thread-safe input/output queue system for parallel processing.
    """
    
    def __init__(
        self,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        max_queue_size: int = 10
    ):
        """
        Args:
            preprocess_fn: Function to preprocess inputs (resize, normalize, etc.)
            postprocess_fn: Function to postprocess outputs (denormalize, convert to PIL, etc.)
            max_queue_size: Maximum number of items in queues
        """
        self.preprocess_fn = preprocess_fn or self.default_preprocess
        self.postprocess_fn = postprocess_fn or self.default_postprocess
        
        # Thread-safe queues
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Control flags
        self.running = False
        self.input_thread = None
        self.output_thread = None
        
        # Statistics
        self.frames_preprocessed = 0
        self.frames_postprocessed = 0
        
    @staticmethod
    def default_preprocess(image: Any) -> torch.Tensor:
        """
        Default preprocessing: PIL/numpy → normalized tensor.
        
        Args:
            image: PIL Image or numpy array [H, W, C]
            
        Returns:
            Normalized tensor [C, H, W]
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(image).float() / 127.5 - 1.0
        
        # HWC → CHW
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
            
        return image
        
    @staticmethod
    def default_postprocess(tensor: torch.Tensor) -> Image.Image:
        """
        Default postprocessing: tensor → PIL Image.
        
        Args:
            tensor: Tensor [C, H, W] in range [0, 1]
            
        Returns:
            PIL Image
        """
        # Ensure on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert from [0, 1] to [0, 255]
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        
        # CHW → HWC
        if tensor.ndim == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and PIL
        array = tensor.numpy()
        
        # Handle grayscale
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
        
        image = Image.fromarray(array)
        
        return image
        
    def start(self):
        """Start the input/output processing threads."""
        if self.running:
            return
            
        self.running = True
        
        # Start input processing thread
        self.input_thread = threading.Thread(
            target=self._input_worker,
            daemon=True
        )
        self.input_thread.start()
        
        # Start output processing thread
        self.output_thread = threading.Thread(
            target=self._output_worker,
            daemon=True
        )
        self.output_thread.start()
        
    def stop(self):
        """Stop the processing threads."""
        self.running = False
        
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        if self.output_thread:
            self.output_thread.join(timeout=1.0)
            
    def _input_worker(self):
        """Worker thread for input preprocessing."""
        while self.running:
            try:
                # This would be fed by external input source
                # For now, just a placeholder
                time.sleep(0.001)
            except Exception as e:
                print(f"Input worker error: {e}")
                
    def _output_worker(self):
        """Worker thread for output postprocessing."""
        while self.running:
            try:
                # Get output from queue
                output = self.output_queue.get(timeout=0.1)
                
                # Postprocess
                processed = self.postprocess_fn(output)
                self.frames_postprocessed += 1
                
                # In real application, this would be sent to display/save
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Output worker error: {e}")
                
    def put_input(self, image: Any, block: bool = True, timeout: float = None):
        """
        Add input to preprocessing queue.
        
        Args:
            image: Input image (PIL, numpy, or tensor)
            block: Whether to block if queue is full
            timeout: Timeout for blocking
        """
        preprocessed = self.preprocess_fn(image)
        self.input_queue.put(preprocessed, block=block, timeout=timeout)
        self.frames_preprocessed += 1
        
    def get_input(self, block: bool = True, timeout: float = None) -> Optional[torch.Tensor]:
        """
        Get preprocessed input from queue.
        
        Args:
            block: Whether to block if queue is empty
            timeout: Timeout for blocking
            
        Returns:
            Preprocessed tensor or None if queue is empty
        """
        try:
            return self.input_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def put_output(self, tensor: torch.Tensor, block: bool = True, timeout: float = None):
        """
        Add output to postprocessing queue.
        
        Args:
            tensor: Output tensor from pipeline
            block: Whether to block if queue is full
            timeout: Timeout for blocking
        """
        self.output_queue.put(tensor, block=block, timeout=timeout)
        
    def get_output(self, block: bool = True, timeout: float = None) -> Optional[Image.Image]:
        """
        Get postprocessed output from queue.
        
        Args:
            block: Whether to block if queue is empty
            timeout: Timeout for blocking
            
        Returns:
            Postprocessed PIL Image or None if queue is empty
        """
        try:
            tensor = self.output_queue.get(block=block, timeout=timeout)
            return self.postprocess_fn(tensor)
        except queue.Empty:
            return None
            
    def get_statistics(self) -> dict:
        """Get queue statistics."""
        return {
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'frames_preprocessed': self.frames_preprocessed,
            'frames_postprocessed': self.frames_postprocessed
        }


def demonstrate_io_queue():
    """Educational demonstration of IO queue system."""
    print("=== IO Queue System Demo ===\n")
    
    io_queue = IOQueue(max_queue_size=5)
    io_queue.start()
    
    print("Adding 10 dummy frames to input queue:")
    for i in range(10):
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        io_queue.put_input(dummy_image, block=False)
        print(f"  Frame {i} added")
        
    time.sleep(0.1)
    
    print(f"\nProcessing frames:")
    for i in range(10):
        tensor = io_queue.get_input(block=True, timeout=1.0)
        if tensor is not None:
            print(f"  Frame {i} retrieved: shape={tensor.shape}")
            
            # Simulate GPU processing
            processed = tensor  # In reality: processed = pipeline(tensor)
            
            # Put to output queue
            io_queue.put_output(processed)
    
    stats = io_queue.get_statistics()
    print(f"\n{'='*50}")
    print(f"Statistics:")
    print(f"  Frames preprocessed: {stats['frames_preprocessed']}")
    print(f"  Frames postprocessed: {stats['frames_postprocessed']}")
    
    io_queue.stop()


if __name__ == "__main__":
    demonstrate_io_queue()
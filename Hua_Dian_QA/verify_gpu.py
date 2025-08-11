import torch
from sentence_transformers import SentenceTransformer
import os

def check_gpu_environment():
    """
    Verifies the GPU environment for PyTorch and sentence-transformers.
    """
    print("--- GPU Environment Verification ---")
    
    # 1. Check PyTorch and CUDA availability
    try:
        print(f"PyTorch version: {torch.__version__}")
        is_cuda_available = torch.cuda.is_available()
        print(f"CUDA available for PyTorch: {is_cuda_available}")
        
        if not is_cuda_available:
            print("\n[ERROR] PyTorch cannot find a compatible CUDA device.")
            print("This is the most likely reason for slow performance.")
            print("Please ensure you have installed the correct PyTorch version for your NVIDIA driver and CUDA toolkit.")
            print("Visit: https://pytorch.org/get-started/locally/")
            return
            
        # 2. Check GPU details
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs found: {gpu_count}")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
        
        # 3. Test loading a sentence-transformer model to GPU
        print("\n--- Testing sentence-transformer model loading ---")
        model_name = "moka-ai/m3e-base"
        print(f"Attempting to load model '{model_name}' onto 'cuda'...")
        try:
            # Suppress download progress bar for cleaner output
            os.environ['TQDM_DISABLE'] = '1'
            model = SentenceTransformer(model_name, device='cuda')
            print("[SUCCESS] Model loaded to GPU successfully.")
            
            # Test encoding
            print("Testing a sample encoding on GPU...")
            test_vector = model.encode("Hello, GPU!", device='cuda')
            print(f"Encoding successful. Vector dimension: {len(test_vector)}")
            del model # Free up memory
            
        except Exception as e:
            print(f"\n[ERROR] Failed to load or use sentence-transformer model on GPU.")
            print(f"Error details: {e}")
            print("This confirms a problem with your environment setup.")

    except ImportError:
        print("\n[ERROR] PyTorch is not installed. Please install it first.")
    
    finally:
        print("\n--- Verification Complete ---")

if __name__ == "__main__":
    check_gpu_environment()

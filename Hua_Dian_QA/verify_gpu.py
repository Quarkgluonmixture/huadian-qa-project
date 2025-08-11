import os
import torch
import logging
import time
from Hua_Dian_QA.data.vector_store import VectorStore
from Hua_Dian_QA.core.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_gpu_availability():
    """Checks for PyTorch, FAISS, and CUDA availability and prints GPU information."""
    logging.info("1. Verifying GPU support for PyTorch and FAISS...")
    
    # Check PyTorch
    try:
        if not torch.cuda.is_available():
            logging.error("PyTorch CUDA is not available. GPU acceleration for embeddings will not work.")
            logging.error("Please install a GPU-enabled version of PyTorch from: https://pytorch.org/get-started/locally/")
            return False
        
        pt_gpu_count = torch.cuda.device_count()
        logging.info(f"PyTorch found {pt_gpu_count} CUDA-enabled GPU(s).")
        for i in range(pt_gpu_count):
            logging.info(f"  - PyTorch GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logging.error("PyTorch is not installed. Please install it with CUDA support.")
        return False
    except Exception as e:
        logging.error(f"An error occurred while checking for PyTorch GPU: {e}")
        return False

    # Check FAISS
    try:
        import faiss
        faiss_gpu_count = faiss.get_num_gpus()
        if faiss_gpu_count == 0:
            logging.warning("FAISS-CPU is installed or no GPUs were found by FAISS. Vector search will run on CPU.")
            logging.warning("For GPU-accelerated search, please install 'faiss-gpu'.")
        else:
            logging.info(f"FAISS found {faiss_gpu_count} GPU(s).")
    except ImportError:
        logging.error("FAISS is not installed. Please install 'faiss-gpu' for GPU support or 'faiss-cpu' for CPU support.")
        return False
    except Exception as e:
        logging.error(f"An error occurred while checking for FAISS GPU: {e}")
        return False
        
    return True

def test_embedding_on_gpu():
    """Tests if the embedding model and FAISS run on the GPU."""
    logging.info("\n2. Testing Embedding and Vector Store (FAISS) on GPU...")
    try:
        vs = VectorStore()
        device = vs.embeddings.client.device
        logging.info(f"HuggingFaceEmbeddings is configured to use device: '{device}'")
        
        if 'cuda' not in str(device):
            logging.warning("Embedding model is NOT configured for CUDA. This is a critical performance issue.")
        else:
            logging.info("Embedding model is correctly configured for GPU.")

        # Create a dummy document and test embedding with FAISS
        logging.info("Creating a sample document and building FAISS index...")
        start_time = time.time()
        
        # Create a dummy document file if it doesn't exist
        from Hua_Dian_QA.core.config import DOCS_PATH
        if not os.path.exists(DOCS_PATH):
            os.makedirs(DOCS_PATH)
        dummy_doc_path = os.path.join(DOCS_PATH, "verify_gpu_sample.txt")
        with open(dummy_doc_path, "w", encoding="utf-8") as f:
            f.write("This is a test document for FAISS and GPU verification.")
        
        documents = vs.load_documents()
        vs.create_vector_store(documents)
        retriever = vs.get_retriever()
        
        if not retriever:
            logging.error("Failed to create or load the retriever. Aborting test.")
            return None

        retrieved_docs = retriever.get_relevant_documents("test verification")
        end_time = time.time()
        
        logging.info(f"FAISS index creation and retrieval test completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Successfully retrieved {len(retrieved_docs)} documents.")
        return retriever
    except Exception as e:
        logging.error(f"An error occurred during the embedding/FAISS test: {e}", exc_info=True)
        return None

def test_llm_on_gpu(retriever):
    """Tests if the LLM (Ollama) utilizes the GPU."""
    if not retriever:
        logging.error("Retriever not available. Skipping LLM test.")
        return

    logging.info("\n3. Testing LLM (Ollama) GPU Acceleration...")
    try:
        llm_manager = LLMManager(retriever)
        
        if llm_manager.llm.num_gpu > 0:
            logging.info(f"Ollama is configured with num_gpu={llm_manager.llm.num_gpu}. It will attempt to use the GPU.")
            logging.info("To confirm usage, monitor your GPU activity using 'nvidia-smi' in a separate terminal while this script is running.")
        else:
            logging.warning("Ollama is configured with num_gpu=0. It will run on the CPU.")

        logging.info("Sending a test query to the RAG chain...")
        question = "What is this document about?"
        chat_history = []
        
        start_time = time.time()
        response = llm_manager.answer_question(question, chat_history)
        end_time = time.time()
        
        logging.info(f"LLM query completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Response from LLM: {response.get('answer', 'No answer found.')}")
        logging.info("LLM test finished. Check your GPU usage to confirm acceleration.")

    except Exception as e:
        logging.error(f"An error occurred during the LLM test: {e}")

if __name__ == '__main__':
    if check_gpu_availability():
        retriever = test_embedding_on_gpu()
        test_llm_on_gpu(retriever)
    logging.info("\nGPU verification script finished.")

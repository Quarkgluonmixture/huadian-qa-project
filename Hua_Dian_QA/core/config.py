import os

# --- Configuration ---
# Get the absolute path of the directory where the script is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(APP_DIR, "..", "docs")
VECTORSTORE_PATH = os.path.join(APP_DIR, "..", "vectorstore")
# EMBEDDING_MODEL = "nomic-embed-text" # GPU model
# EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5" # Original model, incompatible due to missing safetensors.
EMBEDDING_MODEL = "moka-ai/m3e-base" # High-quality alternative with safetensors support.
LLM_MODEL = "huadian-llm" # Use custom, GPU-optimized model
DEVICE = "cuda" # Switch to GPU mode
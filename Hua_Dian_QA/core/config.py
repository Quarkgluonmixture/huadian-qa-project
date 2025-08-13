import os

# --- Configuration ---
# Get the absolute path of the directory where the script is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
DOCS_PATH = os.path.join(PROJECT_ROOT, "docs")
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "vectorstore")
# EMBEDDING_MODEL = "nomic-embed-text" # GPU model
# EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5" # Original model, incompatible due to missing safetensors.
EMBEDDING_MODEL = os.path.join(PROJECT_ROOT, "..", "m3e-base") # Use local model
LLM_MODEL = "huadian-llm" # Use custom, GPU-optimized model
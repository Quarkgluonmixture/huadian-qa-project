import os

# --- Configuration ---
# Get the absolute path of the directory where the script is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(APP_DIR, "..", "docs")
VECTORSTORE_PATH = os.path.join(APP_DIR, "..", "vectorstore")
# EMBEDDING_MODEL = "nomic-embed-text" # GPU model
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5" # Lighter Hugging Face model
LLM_MODEL = "deepseek-llm:7b-chat" # GPU model
DEVICE = "cuda" # Switch to GPU mode
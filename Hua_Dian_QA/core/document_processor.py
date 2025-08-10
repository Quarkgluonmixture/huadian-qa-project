import os
import glob
import time
import logging
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def ensure_docs_directory_exists(docs_path):
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    return os.path.exists(docs_path) and any(fname.endswith(('.txt', '.pdf', '.docx')) for fname in os.listdir(docs_path))

def load_documents_with_error_handling(path):
    logger.info(f"开始从目录 {path} 加载文档...")
    start_time = time.time()
    
    loader_map = {
        ".txt": (TextLoader, {'encoding': 'utf-8'}),
        ".pdf": (PyMuPDFLoader, {}),
    }
    
    all_docs = []
    file_paths = []
    file_paths.extend(glob.glob(f"{path}/**/*.txt", recursive=True))
    file_paths.extend(glob.glob(f"{path}/**/*.pdf", recursive=True))
    file_paths.extend(glob.glob(f"{path}/**/*.docx", recursive=True))

    for file_path in file_paths:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == ".docx":
            try:
                from unstructured.documents.elements import Text
                from unstructured.partition.docx import partition_docx
                
                elements = partition_docx(filename=file_path)
                text_content = "\n".join([str(el) for el in elements if isinstance(el, Text)])
                
                doc = Document(page_content=text_content, metadata={'source': file_path})
                all_docs.append(doc)
            except ImportError:
                logger.warning(f"Skipping file {file_path} due to unstructured package not found. Please install it with `pip install unstructured`")
            except Exception as e:
                logger.error(f"Skipping file {file_path} due to error: {e}")
        elif ext in loader_map:
            loader_cls, kwargs = loader_map[ext]
            try:
                loader = loader_cls(file_path, **kwargs)
                all_docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Skipping file {os.path.basename(file_path)} due to error: {e}")

    end_time = time.time()
    logger.info(f"文档加载完成，总共找到 {len(all_docs)} 个文档，耗时: {end_time - start_time:.2f} 秒。")
    return all_docs

def list_docs(docs_path):
    if not os.path.exists(docs_path):
        return "知识库目录未找到。"
    files = [f for f in os.listdir(docs_path) if f.endswith(('.txt', '.pdf', '.docx'))]
    return "\n".join(files) if files else "未找到任何文档。"

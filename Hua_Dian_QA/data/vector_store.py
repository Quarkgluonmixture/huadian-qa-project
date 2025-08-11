import os
import time
import logging
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from Hua_Dian_QA.core.config import VECTORSTORE_PATH, EMBEDDING_MODEL, DOCS_PATH
from Hua_Dian_QA.core.document_processor import load_documents_with_error_handling

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, store_path=VECTORSTORE_PATH, embedding_model_name=EMBEDDING_MODEL, table_name="huadianqa"):
        self.store_path = store_path
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        
        logger.info(f"Loading embedding model '{self.embedding_model_name}'...")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cuda')
        except Exception as e:
            logger.warning(f"Failed to load model directly, trying with trust_remote_code=True. Error: {e}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cuda', trust_remote_code=True)

        logger.info(f"Connecting to LanceDB at {self.store_path}...")
        self.db = lancedb.connect(self.store_path)
        self.table = None

    def table_exists(self):
        """Checks if the LanceDB table exists."""
        return self.table_name in self.db.table_names()

    def load_documents(self, doc_path=DOCS_PATH):
        """Loads documents from a directory."""
        return load_documents_with_error_handling(doc_path)

    def create_vector_store(self, documents):
        """Creates a LanceDB vector store from documents with GPU acceleration."""
        if not documents:
            logger.warning("No documents found to create a vector store.")
            return None

        logger.info("Processing documents into a DataFrame...")
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        df = pd.DataFrame({"text": texts, "metadata": metadata})

        logger.info(f"Generating embeddings for {len(df)} text chunks...")
        embeddings = self.embedding_model.encode(df["text"].tolist(), show_progress_bar=True, device='cuda')
        
        # LanceDB expects a list of lists/vectors, not a single numpy array
        df["vector"] = [list(emb) for emb in embeddings]

        logger.info(f"Starting to create a new LanceDB table '{self.table_name}'...")
        start_time = time.time()
        
        if self.table_name in self.db.table_names():
            logger.warning(f"Table '{self.table_name}' already exists. Dropping it.")
            self.db.drop_table(self.table_name)

        self.table = self.db.create_table(self.table_name, data=df, mode="overwrite")
        
        logger.info("Creating GPU-accelerated index (IVF_PQ)...")
        self.table.create_index(num_partitions=256, num_sub_vectors=16, accelerator="cuda")

        end_time = time.time()
        logger.info(f"LanceDB table and GPU index creation completed in {end_time - start_time:.2f} seconds.")
        return self.table

    def get_retriever(self, k=5):
        """Returns a function that can be used as a retriever."""
        if self.table is None:
            if self.table_name not in self.db.table_names():
                logger.error(f"LanceDB table '{self.table_name}' does not exist.")
                return None
            logger.info(f"Opening existing LanceDB table '{self.table_name}'.")
            self.table = self.db.open_table(self.table_name)

        def retrieve(query_text):
            logger.info(f"Searching for: '{query_text}'")
            query_vector = self.embedding_model.encode(query_text, device='cuda')
            results = self.table.search(query_vector).limit(k).to_pandas()
            # Reconstruct Document objects for compatibility if needed elsewhere
            # Reconstruct Document objects for compatibility with LangChain chains
            retrieved_docs = [
                Document(page_content=row["text"], metadata=row["metadata"])
                for index, row in results.iterrows()
            ]
            return retrieved_docs

        return RunnableLambda(retrieve)

if __name__ == '__main__':
    # Example usage
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        with open(os.path.join(DOCS_PATH, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("这是一个使用LanceDB和原生SentenceTransformer的示例文档。")

    vs = VectorStore()
    documents = vs.load_documents()
    if documents:
        vs.create_vector_store(documents)
        retriever = vs.get_retriever()
        if retriever:
            print("LanceDB vector store created and retriever is ready.")
            retrieved_docs = retriever("LanceDB示例")
            print(retrieved_docs)
        else:
            print("Failed to create or load retriever.")
    else:
        print("No documents found to process.")
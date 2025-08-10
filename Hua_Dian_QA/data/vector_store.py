import os
import time
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from Hua_Dian_QA.core.config import VECTORSTORE_PATH, EMBEDDING_MODEL, DOCS_PATH
from Hua_Dian_QA.core.document_processor import load_documents_with_error_handling

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory=VECTORSTORE_PATH, embedding_model_name=EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = None

    def load_documents(self, doc_path=DOCS_PATH):
        """Loads documents from a directory using the advanced loader."""
        return load_documents_with_error_handling(doc_path)

    def create_vector_store(self, documents):
        """Creates a vector store from documents."""
        if not documents:
            logger.warning("No documents found to create vector store.")
            # Initialize an empty vector store
            self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
            self.vector_store.persist()
            return self.vector_store

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        logger.info("Starting to create a new vectorstore in batches...")
        start_time = time.time()
        batch_size = 1000
        total_splits = len(docs)
        
        # Initialize the vector store with the first batch
        self.vector_store = Chroma.from_documents(docs[:batch_size], self.embeddings, persist_directory=self.persist_directory)

        # Add remaining batches
        with tqdm(total=total_splits, desc="Creating Vectorstore", unit="split") as pbar:
            pbar.update(min(batch_size, total_splits))
            for i in range(batch_size, total_splits, batch_size):
                batch_end = min(i + batch_size, total_splits)
                batch = docs[i:batch_end]
                self.vector_store.add_documents(batch)
                pbar.update(len(batch))

        self.vector_store.persist()
        end_time = time.time()
        logger.info(f"Vectorstore creation completed in {end_time - start_time:.2f} seconds.")
        return self.vector_store

    def get_retriever(self):
        """Gets the retriever from the vector store."""
        if self.vector_store is None:
            if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
                 logger.warning(f"Vector store at {self.persist_directory} is empty or does not exist.")
                 # Create an empty store to avoid errors, but it won't be useful
                 self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
            else:
                self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return self.vector_store.as_retriever()

if __name__ == '__main__':
    # Example usage
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        with open(os.path.join(DOCS_PATH, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("这是一个示例文档。")

    vs = VectorStore()
    documents = vs.load_documents()
    vs.create_vector_store(documents)
    retriever = vs.get_retriever()
    print("Vector store created and retriever is ready.")
    retrieved_docs = retriever.get_relevant_documents("示例")
    print(retrieved_docs)
import os
import time
import logging
import lancedb
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from hashlib import md5
import json
from Hua_Dian_QA.core.config import VECTORSTORE_PATH, EMBEDDING_MODEL, DOCS_PATH
from Hua_Dian_QA.core.document_processor import load_documents_with_error_handling

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, db_path=VECTORSTORE_PATH, table_name="huadianqa", embedding_model_name=EMBEDDING_MODEL):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        self.doc_dir = DOCS_PATH
        self.metadata_file = os.path.join(self.db_path, 'doc_metadata.json')

        # Check for CUDA availability and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device.upper()}")

        logger.info(f"Loading embedding model '{self.embedding_model_name}' on {self.device.upper()}...")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        except Exception as e:
            logger.warning(f"Failed to load model directly, trying with trust_remote_code=True. Error: {e}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device, trust_remote_code=True)

        logger.info(f"Connecting to LanceDB at {self.db_path}...")
        self.db = lancedb.connect(self.db_path)
        self.table = self.db.open_table(self.table_name) if self.table_name in self.db.table_names() else None

    def _load_doc_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_doc_metadata(self, metadata):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _get_doc_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return md5(f.read()).hexdigest()

    def _split_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)

    def build_or_update(self, force_rebuild=False):
        logger.info("Starting knowledge base build or update process...")
        existing_metadata = self._load_doc_metadata()
        new_metadata = {}
        docs_to_add = []
        docs_to_delete = []

        current_files = {f for f in os.listdir(self.doc_dir) if os.path.isfile(os.path.join(self.doc_dir, f)) and f.endswith(('.txt', '.pdf', '.docx'))}
        existing_files = set(existing_metadata.keys())

        # Find new and modified files
        for file in current_files:
            file_path = os.path.join(self.doc_dir, file)
            file_hash = self._get_doc_hash(file_path)
            new_metadata[file] = file_hash
            if file not in existing_files or existing_metadata[file] != file_hash:
                docs_to_add.append(file_path)

        # Find deleted files
        deleted_files = existing_files - current_files
        if deleted_files:
            docs_to_delete.extend(list(deleted_files))

        if force_rebuild:
            logger.info("Forcing a full rebuild of the vector store.")
            if self.table_name in self.db.table_names():
                self.db.drop_table(self.table_name)
            self.table = None
            all_docs_paths = [os.path.join(self.doc_dir, f) for f in current_files]
            docs = load_documents_with_error_handling(self.doc_dir, file_paths=all_docs_paths)
            chunks = self._split_docs(docs)
            if chunks:
                self._create_table_from_chunks(chunks)
        else:
            # Delete records of deleted files
            if docs_to_delete and self.table:
                delete_query = " or ".join([f"metadata.source like '%{os.path.basename(f)}%'" for f in docs_to_delete])
                logger.info(f"Deleting records for: {docs_to_delete}")
                self.table.delete(delete_query)

            # Add new and modified files
            if docs_to_add:
                logger.info(f"Adding or updating records for: {docs_to_add}")
                docs = load_documents_with_error_handling(self.doc_dir, file_paths=docs_to_add)
                chunks = self._split_docs(docs)
                if chunks:
                    if self.table is None:
                        self._create_table_from_chunks(chunks)
                    else:
                        self._add_chunks_to_table(chunks)
        
        if docs_to_add or docs_to_delete or force_rebuild:
             if self.table:
                logger.info("Creating index for the table...")
                accelerator = "cuda" if self.device == "cuda" else "cpu"
                self.table.create_index(num_partitions=64, num_sub_vectors=8, accelerator=accelerator, replace=True)

        self._save_doc_metadata(new_metadata)
        logger.info("Knowledge base build or update process finished.")
        return True

    def _create_table_from_chunks(self, chunks):
        logger.info(f"Creating new table '{self.table_name}' with {len(chunks)} chunks.")
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=128, device=self.device)
        data = pd.DataFrame({'text': texts, 'metadata': metadata, 'vector': [e.tolist() for e in embeddings]})
        self.table = self.db.create_table(self.table_name, data, mode="overwrite")

    def _add_chunks_to_table(self, chunks):
        logger.info(f"Adding {len(chunks)} new chunks to table '{self.table_name}'.")
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=128, device=self.device)
        data = pd.DataFrame({'text': texts, 'metadata': metadata, 'vector': [e.tolist() for e in embeddings]})
        self.table.add(data)

    def get_retriever(self, k=5):
        if self.table is None:
            if self.table_name not in self.db.table_names():
                logger.error(f"LanceDB table '{self.table_name}' does not exist.")
                return None
            logger.info(f"Opening existing LanceDB table '{self.table_name}'.")
            self.table = self.db.open_table(self.table_name)

        def retrieve(query_text):
            logger.info(f"Searching for: '{query_text}'")
            query_vector = self.embedding_model.encode(query_text, device=self.device)
            results = self.table.search(query_vector).limit(k).to_pandas()
            retrieved_docs = [
                Document(page_content=row["text"], metadata=row["metadata"])
                for _, row in results.iterrows()
            ]
            return retrieved_docs

        return RunnableLambda(retrieve)
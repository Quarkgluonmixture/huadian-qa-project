import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Tuple
from Hua_Dian_QA.core.llm_manager import LLMManager
from Hua_Dian_QA.data.vector_store import VectorStore
from Hua_Dian_QA.core.config import DOCS_PATH

app = FastAPI()
logger = logging.getLogger(__name__)

def initialize_rag_components():
    """Initializes or re-initializes the RAG components."""
    logger.info("Initializing RAG components...")
    try:
        vector_store = VectorStore()
        # If the vector store table doesn't exist, build it now.
        if vector_store.table is None:
            logger.warning("Vector store table not found. Starting initial build.")
            vector_store.build_or_update()
            # Re-initialize to load the newly created table
            vector_store = VectorStore()
        
        retriever = vector_store.get_retriever()
        llm_manager = LLMManager(retriever)
        
        app.state.vector_store = vector_store
        app.state.llm_manager = llm_manager
        logger.info("RAG components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        app.state.vector_store = None
        app.state.llm_manager = None

@app.on_event("startup")
async def startup_event():
    """
    Application startup event. Initializes RAG components in the background.
    """
    # Run initialization in a background task to not block the startup
    # In a real-world scenario, you might use a more robust solution like Celery.
    # For this project, a simple background task is sufficient.
    # Note: FastAPI startup events must complete before the server is operational.
    # So we do it synchronously here, but the logic is now self-contained.
    initialize_rag_components()


class Question(BaseModel):
    text: str
    history: List[Tuple[str, str]] = []

class DeleteFilesRequest(BaseModel):
    filenames: List[str]

@app.post("/ask")
def ask_question(question: Question, request: Request):
    """
    Handles a question from the user, using the LLMManager initialized at startup.
    """
    llm_manager = request.app.state.llm_manager
    if not llm_manager:
        raise HTTPException(status_code=503, detail="RAG service is not available due to initialization failure.")

    # Convert history to the format expected by the LLMManager
    chat_history = []
    for user_msg, ai_msg in question.history:
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": ai_msg})
        
    answer = llm_manager.answer_question(question.text, chat_history)
    # Return both the answer and the retrieved context for debugging
    retrieved_docs = [doc.page_content for doc in answer["context"]]
    return {
        "answer": answer["answer"],
        "source_documents": answer["context"],
        "retrieved_context": retrieved_docs
    }

@app.get("/files")
def list_files():
    """列出 docs 目录下的所有文件。"""
    if not os.path.exists(DOCS_PATH):
        return []
    return [f for f in os.listdir(DOCS_PATH) if os.path.isfile(os.path.join(DOCS_PATH, f))]

@app.post("/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传一个或多个文件到 docs 目录。"""
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
    
    uploaded_filenames = []
    for file in files:
        file_path = os.path.join(DOCS_PATH, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not upload file: {file.filename}. Error: {e}")
        finally:
            file.file.close()
            
    return {"message": f"Successfully uploaded {len(uploaded_filenames)} files.", "filenames": uploaded_filenames}

@app.get("/health")
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}

@app.post("/files/delete")
def delete_files(request: DeleteFilesRequest):
    """删除 docs 目录中指定的多个文件。"""
    deleted_files = []
    not_found_files = []
    
    for filename in request.filenames:
        file_path = os.path.join(DOCS_PATH, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(filename)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not delete file: {filename}. Error: {e}")
        else:
            not_found_files.append(filename)
            
    if not_found_files:
        raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(not_found_files)}")

    return {"message": f"Successfully deleted {len(deleted_files)} files.", "deleted_files": deleted_files}

def update_knowledge_base_task(force_rebuild: bool):
    """Task to update the knowledge base and reload RAG components."""
    logger.info(f"Starting knowledge base update (force_rebuild={force_rebuild})...")
    try:
        # Use the existing vector_store instance from app.state if available
        vs = app.state.vector_store
        if not vs:
            vs = VectorStore() # Or handle error
        
        vs.build_or_update(force_rebuild=force_rebuild)
        logger.info("Knowledge base update finished. Reloading RAG components.")
        
        # Reload all components to ensure they use the new vector store
        initialize_rag_components()
        logger.info("RAG components reloaded successfully after update.")
        
    except Exception as e:
        logger.error(f"Failed to update knowledge base: {e}", exc_info=True)

@app.post("/kb/update")
async def update_kb(background_tasks: BackgroundTasks):
    """Triggers an incremental update of the knowledge base in the background."""
    background_tasks.add_task(update_knowledge_base_task, force_rebuild=False)
    return {"message": "知识库增量更新已开始，请稍后查看状态。"}

@app.post("/kb/rebuild")
async def rebuild_kb(background_tasks: BackgroundTasks):
    """Triggers a full rebuild of the knowledge base in the background."""
    background_tasks.add_task(update_knowledge_base_task, force_rebuild=True)
    return {"message": "知识库强制重建已开始，请稍后查看状态。"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
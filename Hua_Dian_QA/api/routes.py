from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from Hua_Dian_QA.core.llm_manager import LLMManager
from Hua_Dian_QA.data.vector_store import VectorStore

app = FastAPI()

# Initialize components
vs = VectorStore()
retriever = vs.get_retriever()
llm_manager = LLMManager(retriever)

class Question(BaseModel):
    text: str
    history: List[Tuple[str, str]] = []

@app.post("/ask")
def ask_question(question: Question):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
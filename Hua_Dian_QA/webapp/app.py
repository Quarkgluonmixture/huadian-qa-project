import gradio as gr
import os
import logging
from threading import Thread
import time
from langchain_core.messages import HumanMessage, AIMessage
from Hua_Dian_QA.core.llm_manager import LLMManager
from Hua_Dian_QA.data.vector_store import VectorStore
from Hua_Dian_QA.core.config import DOCS_PATH

logger = logging.getLogger(__name__)

class WebApp:
    def __init__(self):
        self.llm_manager = None
        self.vector_store = VectorStore()
        self.init_rag_thread()

    def init_rag_thread(self):
        """Initializes the RAG chain in a background thread."""
        logger.info("Initializing RAG engine in a background thread...")
        # Initial check for the vector store table
        if self.vector_store.table is None:
            logger.warning("Vector store table not found. Starting initial build in background.")
            # Run the build in a background thread to not block the UI
            thread = Thread(target=self.update_knowledge_base_background)
            thread.start()
        else:
            logger.info("Vector store found. Initializing LLMManager.")
            self.initialize_llm_manager()

    def initialize_llm_manager(self):
        """Initializes the LLM Manager with the current vector store's retriever."""
        retriever = self.vector_store.get_retriever()
        if retriever:
            self.llm_manager = LLMManager(retriever)
            logger.info("RAG engine initialized successfully.")
        else:
            logger.error("Failed to initialize RAG engine: Could not get retriever.")

    def update_knowledge_base_background(self, force_rebuild=False, progress=gr.Progress()):
        """Runs the knowledge base update in the background and re-initializes the RAG chain."""
        if progress:
            progress(0, desc="开始更新知识库...")
        
        try:
            self.vector_store.build_or_update(force_rebuild=force_rebuild)
            # Hot-reload: create a new VectorStore instance to load the updated table
            self.vector_store = VectorStore()
            self.initialize_llm_manager()
            status_message = "知识库更新完成！"
            logger.info(status_message)
        except Exception as e:
            status_message = f"知识库更新失败: {e}"
            logger.exception(status_message)
        
        if progress:
            progress(1, desc=status_message)
        
        # This part is tricky with Gradio's state management.
        # We return the message, and the UI will have to handle it.
        return status_message

    def get_answer(self, question, history):
        if self.llm_manager is None:
            return "问答引擎正在初始化中，请稍候..."

        chat_history = [HumanMessage(content=user_msg) for user_msg, _ in history]
        chat_history.extend([AIMessage(content=ai_msg) for _, ai_msg in history])

        response = self.llm_manager.answer_question(question, chat_history)
        answer = response.get("answer", "抱歉，我无法回答这个问题。")
        
        source_documents = response.get("context", [])
        unique_sources = {os.path.basename(doc.metadata.get('source', '')) for doc in source_documents}
        if unique_sources:
            sources_text = "\n\n---\n*参考来源:*\n" + "\n".join(f"- `{s}`" for s in sorted(list(unique_sources)))
            answer += sources_text
            
        return answer

    def launch(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 华电问答机器人")

            with gr.Tab("智能问答"):
                chatbot = gr.Chatbot(label="聊天窗口", avatar_images=(None, "https://img.icons8.com/color/48/000000/robot-2.png"))
                msg = gr.Textbox(label="请输入您的问题...", placeholder="在这里输入问题然后按 enter 键")
                clear = gr.Button("🧹 清除对话")

                def respond(message, chat_history):
                    bot_message = self.get_answer(message, chat_history)
                    chat_history.append((message, bot_message))
                    return "", chat_history

                msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
                clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

            with gr.Tab("知识库管理"):
                update_status = gr.Textbox(label="更新状态", interactive=False)
                
                with gr.Row():
                    update_btn = gr.Button("增量更新知识库")
                    force_rebuild_btn = gr.Button("强制重建知识库")

                update_btn.click(
                    lambda: self.update_knowledge_base_background(force_rebuild=False),
                    outputs=update_status
                )
                force_rebuild_btn.click(
                    lambda: self.update_knowledge_base_background(force_rebuild=True),
                    outputs=update_status
                )

            demo.queue().launch(server_name="127.0.0.1", share=True)
import gradio as gr
import os
import logging
from langchain_core.messages import HumanMessage, AIMessage
from Hua_Dian_QA.core.llm_manager import LLMManager
from Hua_Dian_QA.data.vector_store import VectorStore
from Hua_Dian_QA.core.config import DOCS_PATH

# Global variables
llm_manager = None
logger = logging.getLogger(__name__)

def list_docs(doc_path):
    """Lists documents in the specified directory."""
    if not os.path.exists(doc_path):
        return []
    return [f for f in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, f))]

def delete_files(files_to_delete):
    """Deletes selected files from the docs directory."""
    if not files_to_delete:
        return "未选择任何文件。"
    
    for filename in files_to_delete:
        file_path = os.path.join(DOCS_PATH, filename)
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return f"删除文件 {filename} 时出错。"

    return f"成功删除 {len(files_to_delete)} 个文件。建议重建知识库。"

def init_rag(force_rebuild=False):
    """
    Initializes the RAG chain. Can be forced to rebuild the vectorstore.
    """
    global llm_manager
    logger.info("Initializing RAG engine...")
    vs = VectorStore()
    if force_rebuild or not os.path.exists(vs.persist_directory) or not os.listdir(vs.persist_directory):
        logger.info("Rebuilding vector store...")
        documents = vs.load_documents()
        vs.vector_store = vs.create_vector_store(documents)
    
    retriever = vs.get_retriever()
    llm_manager = LLMManager(retriever)
    logger.info("RAG engine initialized.")
    return "问答引擎已就绪。"

def get_answer(question, history):
    """
    Invokes the RAG chain to get an answer for the given question and history.
    """
    if llm_manager is None:
        logger.warning("RAG engine not initialized. Please initialize it first.")
        return "问答引擎未初始化，请先初始化。"

    logger.info(f"Received question: {question}")

    # Convert Gradio history to LangChain history
    chat_history = []
    for user_msg, ai_msg in history:
        chat_history.append(HumanMessage(content=user_msg))
        chat_history.append(AIMessage(content=ai_msg))

    logger.info(f"Chat history contains {len(chat_history)} messages.")

    response = llm_manager.answer_question(question, chat_history)
    answer = response["answer"]
    logger.info(f"Generated answer: {answer}")
    
    source_documents = response.get("context", [])
    unique_sources = set()
    if source_documents:
        for doc in source_documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_path = doc.metadata['source']
                source_filename = os.path.basename(os.path.normpath(source_path))
                unique_sources.add(source_filename)
    
    logger.info(f"Retrieved sources: {unique_sources}")

    if unique_sources:
        sources_text = "\n\n---\n*参考来源:*\n" + "\n".join(f"- `{s}`" for s in sorted(list(unique_sources)))
        answer += sources_text
    
    return answer

def clear_chat():
    """Clears the chatbot and the message input box."""
    return [], ""

def upload_file(files):
    """Saves uploaded files to the docs directory."""
    if not files:
        return "未选择任何文件。"

    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(DOCS_PATH, filename)
        
        # Read the content of the temporary file and write it to the destination
        with open(file.name, 'rb') as f_in:
            content = f_in.read()
        
        with open(dest_path, 'wb') as f_out:
            f_out.write(content)
            
        logger.info(f"Uploaded file '{filename}' to '{dest_path}'.")
        
    return f"成功上传 {len(files)} 个文件。建议重建知识库。"

# --- Custom CSS for visual optimization ---
custom_css = """
body {
    background-color: #f0f2f5;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-container {
    border-radius: 15px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
#rag-title {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    padding: 20px 0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 华电问答机器人", elem_id="rag-title")
    
    with gr.Tab("智能问答"):
        chatbot = gr.Chatbot(
            [],
            avatar_images=(None, "https://img.icons8.com/color/48/000000/robot-2.png"),
            elem_id="chatbot-main",
            label="聊天窗口"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="请输入您的问题...",
                placeholder="在这里输入问题然后按 enter 键",
                lines=1,
                show_label=False,
                scale=4
            )

        clear = gr.Button("🧹 清除对话")

        def respond(message, chat_history):
            bot_message = get_answer(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
        clear.click(clear_chat, None, [chatbot, msg], queue=False)

    with gr.Tab("知识库文档"):
        file_status_msg = gr.Markdown("")
        
        with gr.Row():
            upload_button = gr.UploadButton("📁 上传文件", file_count="multiple")
            delete_btn = gr.Button("🗑️ 删除所选文件")
        
        doc_list = gr.CheckboxGroup(label="可用文档列表", value=lambda: list_docs(DOCS_PATH), interactive=True)
        
        refresh_btn = gr.Button("🔄 刷新文档列表")

        # Actions
        upload_button.upload(
            upload_file,
            upload_button,
            [file_status_msg]
        ).then(
            lambda: gr.update(choices=list_docs(DOCS_PATH)),
            None,
            [doc_list]
        )

        delete_btn.click(
            delete_files,
            [doc_list],
            [file_status_msg]
        ).then(
            lambda: gr.update(choices=list_docs(DOCS_PATH), value=[]),
            None,
            [doc_list]
        )

        refresh_btn.click(lambda: gr.update(choices=list_docs(DOCS_PATH)), None, doc_list)

    with gr.Tab("管理面板"):
        init_status = gr.Textbox(label="引擎状态", interactive=False, value="正在自动初始化...")
        rebuild_btn = gr.Button("🔄 强制重建知识库")
        
        def force_rebuild_rag():
            return init_rag(force_rebuild=True)
            
        rebuild_btn.click(force_rebuild_rag, None, init_status, queue=False)

    def initial_load():
        init_message = init_rag(force_rebuild=False)
        docs = list_docs(DOCS_PATH)
        return init_message, gr.update(choices=docs)

    demo.load(initial_load, None, [init_status, doc_list])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=True)
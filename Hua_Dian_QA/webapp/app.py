import gradio as gr
import os
import logging
import requests
import time
from functools import partial

logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"

class WebApp:
    def __init__(self):
        self.api_is_ready = False

    def check_api_status(self):
        """Checks if the backend API is ready."""
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                self.api_is_ready = True
                return "API is ready."
            else:
                return "API not ready yet..."
        except requests.ConnectionError:
            return "API not ready yet..."

    def get_answer(self, question, history):
        """Gets an answer from the backend API."""
        if not self.api_is_ready:
            # Add a small delay and retry checking API status
            time.sleep(2)
            status = self.check_api_status()
            if not self.api_is_ready:
                 return "问答引擎正在初始化中，请稍候...（正在下载模型，这可能需要一些时间）"

        # Prepare chat history for the API
        api_history = []
        # The history is a flat list of dicts: [{"role": "user", ...}, {"role": "assistant", ...}]
        # We need to convert it to a list of tuples for the API.
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                user_msg = history[i]["content"]
                ai_msg = history[i+1]["content"]
                api_history.append((user_msg, ai_msg))

        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"text": question, "history": api_history}
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "抱歉，我无法回答这个问题。")
            source_documents = data.get("source_documents", [])
            
            unique_sources = {os.path.basename(doc.get('metadata', {}).get('source', '')) for doc in source_documents}
            if unique_sources:
                sources_text = "\n\n---\n*参考来源:*\n" + "\n".join(f"- `{s}`" for s in sorted(list(unique_sources)) if s)
                answer += sources_text
                
            return answer
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"请求后端服务失败: {e}"

    def update_knowledge_base(self, force_rebuild=False):
        """Sends a request to the API to update the knowledge base."""
        endpoint = "/kb/rebuild" if force_rebuild else "/kb/update"
        try:
            response = requests.post(f"{API_URL}{endpoint}")
            response.raise_for_status()
            return response.json().get("message", "操作成功")
        except requests.RequestException as e:
            logger.error(f"知识库更新请求失败: {e}")
            return f"知识库更新请求失败: {e}"

    def launch(self):
        custom_css = """
        .card {
            border: 1px solid #E5E7EB; /* A light gray border */
            border-radius: 8px;       /* Rounded corners */
            padding: 16px;            /* Some padding inside the box */
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); /* A subtle shadow */
        }
        """
        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
            gr.Markdown("# 华电问答机器人")

            # Initial API status check
            api_status_textbox = gr.Textbox(self.check_api_status, every=2, interactive=False, visible=False)
            
            with gr.Tab("智能问答"):
                chatbot = gr.Chatbot(label="聊天窗口", avatar_images=(None, "https://img.icons8.com/color/48/000000/robot-2.png"), type="messages")
                msg = gr.Textbox(label="请输入您的问题...", placeholder="在这里输入问题然后按 enter 键")
                clear = gr.Button("🧹 清除对话")

                def respond(message, chat_history):
                    bot_message = self.get_answer(message, chat_history)
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_message})
                    return "", chat_history

                msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
                clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

            with gr.Tab("知识库管理") as kb_tab:
                # Helper functions for API calls
                def refresh_file_list():
                    try:
                        response = requests.get(f"{API_URL}/files")
                        response.raise_for_status()
                        files = response.json()
                        return gr.CheckboxGroup(choices=files, value=[], label="知识库文档列表", info="选择文件进行操作")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"获取文件列表失败: {e}")
                        gr.Warning(f"获取文件列表失败: {e}")
                        return gr.CheckboxGroup(choices=[], label="知识库文档列表", info="选择文件进行操作")

                def handle_upload_files(files, progress=gr.Progress()):
                    progress(0, desc="开始上传...")
                    try:
                        upload_files = [("files", (os.path.basename(f.name), open(f.name, "rb"))) for f in files]
                        response = requests.post(f"{API_URL}/files/upload", files=upload_files)
                        response.raise_for_status()
                        progress(1, desc="上传完成！")
                        return response.json().get("message", "上传成功")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"上传文件失败: {e}")
                        return f"上传文件失败: {e}"
                    finally:
                        for _, (_, f) in upload_files:
                            f.close()

                def handle_delete_files(filenames, progress=gr.Progress()):
                    if not filenames:
                        gr.Warning("请先选择要删除的文件！")
                        return "未选择任何文件"
                    progress(0, desc="正在删除...")
                    try:
                        response = requests.post(f"{API_URL}/files/delete", json={"filenames": filenames})
                        response.raise_for_status()
                        progress(1, desc="删除完成！")
                        return response.json().get("message", "删除成功")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"删除文件失败: {e}")
                        return f"删除文件失败: {e}"

                with gr.Column():
                    gr.Markdown("### 📄 文档管理")
                    with gr.Group(elem_classes="card"):
                        file_manager_status = gr.Textbox(label="文件操作状态", interactive=False, lines=1, placeholder="这里将显示文件操作的结果...")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3):
                                file_list_checkbox = gr.CheckboxGroup(label="知识库文档列表", info="选择文件进行操作")
                            with gr.Column(scale=1, min_width=120):
                                refresh_files_btn = gr.Button("🔄 刷新列表")
                        with gr.Row():
                            upload_button = gr.UploadButton("📤 上传文件", file_count="multiple", variant="primary")
                            delete_button = gr.Button("🗑️ 删除选中", variant="stop")
                
                with gr.Column():
                    gr.Markdown("### ⚙️ 知识库更新")
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("对知识库进行增量更新或强制重建。**注意：** 此操作可能需要较长时间，请耐心等待。")
                        update_status = gr.Textbox(label="更新状态", interactive=False, lines=1, placeholder="这里将显示知识库更新的结果...")
                        with gr.Row():
                            update_btn = gr.Button("🔄 增量更新")
                            force_rebuild_btn = gr.Button("💥 强制重建", variant="stop")
                        
                        # Hidden textbox to act as a timer/poller
                        kb_status_poller = gr.Textbox(visible=False)

                # --- Helper function for blocking KB status polling ---
                def start_and_poll_kb_update(force_rebuild):
                    # Step 1: Trigger the update.
                    initial_message = self.update_knowledge_base(force_rebuild=force_rebuild)
                    yield initial_message
                    
                    # Step 2: Poll for the result in a loop.
                    while True:
                        try:
                            response = requests.get(f"{API_URL}/kb/status")
                            response.raise_for_status()
                            status_data = response.json()
                            status = status_data.get("status", "idle")
                            message = status_data.get("message", "")
                            
                            if status != "running":
                                yield message
                                break # Exit the loop
                            
                            # Update the status and wait before the next poll
                            yield message
                            time.sleep(2)

                        except requests.RequestException as e:
                            error_msg = f"无法获取更新状态: {e}"
                            logger.error(f"KB status poll failed: {e}")
                            yield error_msg
                            break # Exit the loop on error

                # --- Component interactions ---
                update_btn.click(
                    partial(start_and_poll_kb_update, force_rebuild=False),
                    outputs=update_status
                )
                force_rebuild_btn.click(
                    partial(start_and_poll_kb_update, force_rebuild=True),
                    outputs=update_status
                )
                
                kb_tab.select(refresh_file_list, None, file_list_checkbox)
                refresh_files_btn.click(refresh_file_list, None, file_list_checkbox)
                
                upload_button.upload(
                    handle_upload_files,
                    inputs=upload_button,
                    outputs=file_manager_status
                ).then(
                    refresh_file_list, None, file_list_checkbox
                )

                delete_button.click(
                    handle_delete_files,
                    inputs=file_list_checkbox,
                    outputs=file_manager_status
                ).then(
                    refresh_file_list, None, file_list_checkbox
                )

            demo.queue().launch(server_name="127.0.0.1", share=True)
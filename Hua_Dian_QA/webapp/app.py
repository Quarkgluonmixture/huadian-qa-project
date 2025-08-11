import gradio as gr
import os
import logging
import requests
import time

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
        for user_msg, ai_msg in history:
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
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
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
                with gr.Blocks():
                    gr.Markdown("## 知识库更新")
                    update_status = gr.Textbox(label="更新状态", interactive=False)
                    
                    with gr.Row():
                        update_btn = gr.Button("增量更新知识库")
                        force_rebuild_btn = gr.Button("强制重建知识库")

                    update_btn.click(
                        lambda: self.update_knowledge_base(force_rebuild=False),
                        outputs=update_status
                    )
                    force_rebuild_btn.click(
                        lambda: self.update_knowledge_base(force_rebuild=True),
                        outputs=update_status
                    )

                gr.Markdown("---")
                gr.Markdown("## 文档管理")

                file_manager_status = gr.Textbox(label="文件操作状态", interactive=False)

                with gr.Row():
                    file_list_checkbox = gr.CheckboxGroup(label="文档列表", info="选择要删除的文件")
                    refresh_files_btn = gr.Button("🔄 刷新")

                with gr.Row():
                    upload_button = gr.UploadButton("上传文件", file_count="multiple")
                    delete_button = gr.Button("🗑️ 删除选中文件")

                # Helper functions for API calls
                def refresh_file_list():
                    try:
                        response = requests.get(f"{API_URL}/files")
                        response.raise_for_status()
                        files = response.json()
                        return gr.CheckboxGroup(choices=files, value=[], label="文档列表", info="选择要删除的文件")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"获取文件列表失败: {e}")
                        gr.Warning(f"获取文件列表失败: {e}")
                        return gr.CheckboxGroup(choices=[], label="文档列表", info="选择要删除的文件")

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
                        # Close the files
                        for _, (_, f) in upload_files:
                            f.close()


                def handle_delete_files(filenames, progress=gr.Progress()):
                    if not filenames:
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

                # Component interactions
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
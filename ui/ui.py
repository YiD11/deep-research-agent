import gradio as gr

from core.rag import RagSystem


def create_gradio_ui():
    rag_system = RagSystem()

    def handle_get_file_list():
        files = rag_system.get_markdown_files()
        if not files:
            return "📭 No documents available in the knowledge base"
        return "\n".join([f"{f}" for f in files])

    def handle_upload(files, progress=gr.Progress()):
        if not files:
            return None, handle_get_file_list()

        added, skipped = rag_system.add_documents(files, progress)
        gr.Info(f"✅ Added: {added} | Skipped: {skipped}")
        return None, handle_get_file_list()

    def handle_clean():
        rag_system.clean_all_documents()
        gr.Info(f"🗑️ Removed all documents")
        return handle_get_file_list()

    async def handle_chat(msg, hist):
        return await rag_system.chat(msg, hist)

    def clean_session():
        rag_system.clean_all_documents()

    with gr.Blocks(title="Agentic RAG") as demo:

        with gr.Tab("Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add New Documents")
            gr.Markdown(
                "Upload PDF or Markdown files. Duplicates will be automatically skipped."
            )

            files_input = gr.File(
                label="Drop PDF or Markdown files here",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False,
            )

            add_btn = gr.Button("Add Documents", variant="primary", size="md")

            gr.Markdown("## Current Documents in the Knowledge Base")
            gr.Markdown(f"Markdown Directory: {rag_system.markdown_dir.absolute()}")
            file_list = gr.Textbox(
                value=handle_get_file_list(),
                interactive=False,
                lines=7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False,
            )

            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")

            add_btn.click(
                handle_upload,
                [files_input],
                [files_input, file_list],
                show_progress="corner",
            )
            refresh_btn.click(handle_get_file_list, None, file_list)
            clear_btn.click(handle_clean, None, file_list)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=600,
                placeholder="Ask me anything about your documents!",
                show_label=False,
            )
            chatbot.clear(clean_session)

            gr.ChatInterface(fn=handle_chat, chatbot=chatbot)

    return demo

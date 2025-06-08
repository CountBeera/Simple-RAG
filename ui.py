# ui.py

import gradio as gr
from chatbot import chat

def respond(user_message, chat_history):
    """Handle a new user message and update the chat history."""
    bot_response = chat(user_message)
    chat_history = chat_history or []
    chat_history.append((user_message, bot_response))
    return chat_history, chat_history

# 1️⃣ Use Gradio’s built-in Dark theme
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align:center; padding:10px 0;">
            <h2 style="margin:0; color:#FFD700;">🕶️ Dark-Mode RAG Chatbot</h2>
            <p style="color:#aaa;">Ask anything about your ingested docs</p>
        </div>
        """,
        elem_id="header"
    )

    chatbot_ui = gr.Chatbot(elem_id="chatbot", label="Chat")

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your question here…",
            show_label=False,
            lines=1,
            elem_id="input-box"
        )
        send_btn = gr.Button("Send", elem_id="send-btn")

    # Wire up interactions
    send_btn.click(respond, [user_input, chatbot_ui], [chatbot_ui, chatbot_ui])
    user_input.submit(respond, [user_input, chatbot_ui], [chatbot_ui, chatbot_ui])

    # Clear button
    clear_btn = gr.Button("Clear chat", elem_id="clear-btn", variant="secondary")
    clear_btn.click(lambda: None, None, chatbot_ui)

if __name__ == "__main__":
    # Launch on localhost:7860 in dark mode
    # demo.launch()
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,   # hide the raw API tab
        share=False
    )

# tech_tutor/src/app.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from tech_tutor.src.tech_tutor import build_messages


def tutor_agent_stream(
    history,
    code,
    favourite_movie,
    provider,
    openai_model,
    ollama_model,
    temperature,
):
    load_dotenv(override=True)

    history = history or []
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    last_user = (last_user or "").strip()

    messages = build_messages(
        question=last_user,
        code=code if (code and code.strip()) else None,
        favourite_movie=favourite_movie,
    )

    # Prepare streaming assistant message
    history = history + [{"role": "assistant", "content": ""}]
    acc = ""

    # Stream text
    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")
        client = OpenAI(api_key=api_key)

        stream = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                acc += delta
                history[-1]["content"] = acc
                yield history

    else:  # Ollama
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        try:
            stream = client.chat.completions.create(
                model=ollama_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    acc += delta
                    history[-1]["content"] = acc
                    yield history
        except Exception:
            # Fallback: one-shot if Ollama streaming isn't available/configured
            resp = client.chat.completions.create(
                model=ollama_model,
                messages=messages,
                temperature=temperature,
            )
            acc = (resp.choices[0].message.content or "").strip()
            history[-1]["content"] = acc
            yield history


def put_message_in_chatbot(message, history):
    history = history or []
    history = history + [{"role": "user", "content": message}]
    return "", history


with gr.Blocks() as ui:
    gr.Markdown("## Tech Tutor")
    gr.Markdown(
        "Ask a question (optionally paste code). Get a clear analogy-backed explanation."
    )

    # --- CHAT FIRST (primary interaction) ---
    chatbot = gr.Chatbot(height=560, type="messages", label="Chat")
    with gr.Row(equal_height=True):
        message = gr.Textbox(
            placeholder="Type your question…",
            show_label=False,
            lines=3,
            scale=10,
        )
        send = gr.Button("Send", variant="primary", scale=1, min_width=90)

    # --- CONTROLS + CODE IN A 2-COLUMN LAYOUT ---
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Row():
                provider = gr.Dropdown(
                    ["OpenAI", "Ollama"], value="OpenAI", label="Provider"
                )

            with gr.Row():
                favourite_movie = gr.Textbox(
                    value="Lord of the Rings", label="Analogy source"
                )
                temperature = gr.Slider(
                    0, 1.2, value=0.7, step=0.1, label="Temperature"
                )

            with gr.Row():
                openai_model = gr.Textbox(value="gpt-4.1-mini", label="OpenAI model")
                ollama_model = gr.Textbox(value="llama3.2", label="Ollama model")

        with gr.Column(scale=2):
            code = gr.Textbox(
                lines=12,
                label="Optional code",
                placeholder="Paste code here…",
            )

    def _submit(msg, hist):
        return put_message_in_chatbot(msg, hist)

    send.click(
        _submit,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    ).then(
        tutor_agent_stream,
        inputs=[
            chatbot,
            code,
            favourite_movie,
            provider,
            openai_model,
            ollama_model,
            temperature,
        ],
        outputs=[chatbot],
    )

    message.submit(
        _submit,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    ).then(
        tutor_agent_stream,
        inputs=[
            chatbot,
            code,
            favourite_movie,
            provider,
            openai_model,
            ollama_model,
            temperature,
        ],
        outputs=[chatbot],
    )

ui.queue()
ui.launch(inbrowser=True)

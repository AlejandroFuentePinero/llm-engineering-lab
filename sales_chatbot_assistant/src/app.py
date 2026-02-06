import gradio as gr
from sales_chatbot_assistant.src.sales_intake_copilot import sales_assistant_stream

demo = gr.ChatInterface(
    fn=sales_assistant_stream,
    type="messages",
    title="Sales Intake Copilot",
    description="Lightweight lead qualification + internal handoff note.",
)

if __name__ == "__main__":
    demo.launch()

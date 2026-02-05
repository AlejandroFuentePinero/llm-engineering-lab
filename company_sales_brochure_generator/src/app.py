import gradio as gr
from company_sales_brochure_generator.src.company_brochure_generator import (
    brochure_generator_stream,
)

demo = gr.Interface(
    fn=brochure_generator_stream,  # generator => Gradio streams automatically
    title="Brochure Generator",
    inputs=[
        gr.Textbox(label="Company name"),
        gr.Textbox(label="Landing page URL (https://...)"),
    ],
    outputs=gr.Markdown(label="Brochure (Markdown)"),
    examples=[
        ["GitHub", "https://github.com/"],
        ["Portfolio", "https://alejandrofuentepinero.github.io/"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

import torch
import gradio as gr
from transformers import pipeline

# Initialize the summarization pipeline
Text_summary = pipeline("summarization", model="facebook/bart-large-cnn", torch_dtype=torch.bfloat16)

# Define a function to estimate token count from word count
def estimate_tokens(word_count):
    # Approximate tokens as 1.5 times the word count
    return int(word_count * 1.5)

# Define the summarization function
def summary(input, word_count):
    # Convert word count to token count
    max_length = estimate_tokens(word_count)
    min_length = max(10, max_length // 2)  # Set a reasonable minimum length
    output = Text_summary(input, max_length=max_length, min_length=min_length)
    return output[0]['summary_text']

# Close any existing Gradio instances
gr.close_all()

# Set up the Gradio interface
Demo = gr.Interface(
    fn=summary,
    inputs=[
        gr.Textbox(label="Input Text To Summarize", lines=20),
        gr.Slider(
            label="Summary Length (Words Approx.)", 
            minimum=50, maximum=300, step=10, value=130
        )
    ],
    outputs=[gr.Textbox(label="Summarized Text", lines=4)],
    title="Text_Summarize_App",
    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT"
)

# Launch the app with a public link
Demo.launch(share=True)

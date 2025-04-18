import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def generate(prompt, width, height):
    image = pipe(prompt, width=int(width), height=int(height)).images[0]
    return image

gr.Interface(
    fn=generate,
    inputs=["text", gr.Slider(256, 1024, step=64), gr.Slider(256, 1024, step=64)],
    outputs="image",
    title="Offline Image Generator"
).launch()

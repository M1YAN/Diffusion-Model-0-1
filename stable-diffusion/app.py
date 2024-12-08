import gradio as gr
from diffusers import DiffusionPipeline
import torch

# 初始化模型
pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

# 定义图像生成函数
def generate_image(prompt):
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

# 使用 Gradio 创建界面
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Image Generation Demo")
    
    with gr.Row():
        # 文本输入框
        text_input = gr.Textbox(label="Enter your prompt", placeholder="Type something to generate an image...")
        
        # 生成按钮和重新生成按钮
        with gr.Column():
            submit_button = gr.Button("Generate Image")
            reset_button = gr.Button("Reset")
    
    # 图像输出展示
    image_output = gr.Image(label="Generated Image")
    
    # 设置生成按钮的操作
    submit_button.click(fn=generate_image, inputs=text_input, outputs=image_output)
    
    # 设置重置按钮的操作
    reset_button.click(fn=lambda: None, inputs=None, outputs=image_output)

# 启动应用
demo.launch(share=True)
from diffusers import DiffusionPipeline
import os
from tqdm import tqdm

pipeline = DiffusionPipeline.from_pretrained("/openbayes/home/checkpoints/stable-diffusion-v1-4").to("cuda")

prompt = "A photo of valley"


image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
img_name = prompt.replace(" ","_")
    
image.save(f"output/{img_name}.png")
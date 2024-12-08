from diffusers import StableDiffusionPipeline
import torch
import os

pipeline = StableDiffusionPipeline.from_pretrained("/openbayes/home/miyan/works/text_to_image/sd-naruto-model", torch_dtype=torch.float16).to("cuda")

prompt = "valley"
image = pipeline(prompt=prompt).images[0]
os.makedirs("samples", exist_ok=True)
img_name = prompt.replace(" ","_")
image.save(os.path.join("samples", f"{img_name}.png"))

print("Enjoy!")
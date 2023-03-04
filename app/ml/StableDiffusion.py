import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline

device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_id = "runwayml/stable-diffusion-v1-5"

text2imgPipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

text2imgPipe = text2imgPipe.to(device)
img2imgPipe = img2imgPipe.to(device)

text2imgPipe.enable_attention_slicing()
img2imgPipe.enable_attention_slicing()

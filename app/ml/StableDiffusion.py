import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

main_model_id = "runwayml/stable-diffusion-v1-5"
inpainting_model_id = "runwayml/stable-diffusion-inpainting"

text2imgPipe = StableDiffusionPipeline.from_pretrained(main_model_id, torch_dtype=torch.float16).to(
    device)
text2imgPipe.enable_attention_slicing()

img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(main_model_id, torch_dtype=torch.float16).to(
    device)
img2imgPipe.enable_attention_slicing()

inpaintingPipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model_id, torch_dtype=torch.float16).to(
    device)
inpaintingPipe.enable_attention_slicing()

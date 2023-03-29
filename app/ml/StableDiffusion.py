import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, \
    StableDiffusionUpscalePipeline

from app.utils.helpers import get_device_name

device = get_device_name()

main_model_id = "runwayml/stable-diffusion-v1-5"
inpainting_model_id = "runwayml/stable-diffusion-inpainting"
upscaler_model_id = "stabilityai/stable-diffusion-x4-upscaler"

text2imgPipe = StableDiffusionPipeline.from_pretrained(main_model_id, torch_dtype=torch.float16).to(
    device)
text2imgPipe.enable_attention_slicing()

img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(main_model_id, torch_dtype=torch.float16).to(
    device)
img2imgPipe.enable_attention_slicing()

inpaintingPipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model_id, torch_dtype=torch.float16).to(
    device)
inpaintingPipe.enable_attention_slicing()

upscalerPipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_model_id, torch_dtype=torch.float16).to(
    device)

import logging
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionUpscalePipeline

from app.my_secrets import access_token
from app.utils.helpers import get_device_name

device = get_device_name()
logging.info(f"Running on device type {device}")

main_model_id = "runwayml/stable-diffusion-v1-5"
inpainting_model_id = "runwayml/stable-diffusion-inpainting"
upscaler_model_id = "stabilityai/stable-diffusion-x4-upscaler"

text2imgPipe = StableDiffusionPipeline.from_pretrained(main_model_id,
                                                       # torch_dtype=torch.float16
                                                       use_auth_token=access_token,
                                                       ).to(device)
text2imgPipe.enable_attention_slicing()

img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(main_model_id,
                                                             # torch_dtype=torch.float16
                                                             use_auth_token=access_token,
                                                             ).to(device)
img2imgPipe.enable_attention_slicing()

inpaintingPipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model_id,
                                                                # torch_dtype=torch.float16
                                                                use_auth_token=access_token,
                                                                ).to(device)
inpaintingPipe.enable_attention_slicing()

upscalerPipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_model_id,
                                                              # torch_dtype=torch.float16
                                                              use_auth_token=access_token,
                                                              ).to(device)
upscalerPipe.enable_attention_slicing()

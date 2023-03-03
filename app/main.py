import io
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from io import BytesIO
import base64
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

# imports for image2image
import requests
from PIL import Image

# Data Models
from .data_models import TextToImageGenerationParameters

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "mps"

model_id = "runwayml/stable-diffusion-v1-5"

text2imgPipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

text2imgPipe = text2imgPipe.to(device)
img2imgPipe = img2imgPipe.to(device)

text2imgPipe.enable_attention_slicing()
img2imgPipe.enable_attention_slicing()


# text2imgPipe.enable_xformers_memory_efficient_attention()
# img2imgPipe.enable_xformers_memory_efficient_attention()


@app.get("/")
def health_check():
    return {"health_check": "healthy"}


@app.post("/text-to-image")
def generate_text_to_imgage(parameters: TextToImageGenerationParameters):
    print(parameters)

    generator = torch.Generator("cpu").manual_seed(parameters.seed)

    imageArray = []
    for i in range(parameters.num_images_per_prompt):
        image = text2imgPipe(
            prompt=parameters.prompt,
            height=parameters.height,
            width=parameters.height,
            num_inference_steps=parameters.num_inference_steps,
            guidance_scale=parameters.guidance_scale,
            negative_prompt=parameters.negative_prompt,
            generator=generator
        ).images[0]
        image.save(f'testimage{i}.png')

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        imageArray.append(imgstr)

    print(imageArray)

    return imageArray


@app.post("/image-to-image")
async def generate_img_to_image(initial_image: UploadFile = File(...),
                                prompt: str = Form(...),
                                strength: float = Form(...),
                                num_inference_steps: int = Form(...),
                                guidance_scale: float = Form(...),
                                negative_prompt: Optional[str] = Form(""),
                                num_images_per_prompt: int = Form(...),
                                seed: int = Form(...)):
    print({"prompt": prompt,
           "strength": strength,
           "num_inference_steps": num_inference_steps,
           "guidance_scale": guidance_scale,
           "negative_prompt": negative_prompt,
           "num_images_per_prompt": num_images_per_prompt,
           "seed": seed})

    contents = initial_image.file.read()
    init_img = Image.open(BytesIO(contents)).convert("RGB")
    init_img = init_img.resize((768, 512))
    init_img.show()

    generator = torch.Generator().manual_seed(1024)

    test_prompt = "A fantasy landscape, trending on artstation"

    generated_image = \
        img2imgPipe(prompt=test_prompt, image=init_img, strength=0.75, guidance_scale=7.5, generator=generator).images[
            0]

    generated_image.save('testimage2image.png')

    return {"message": f"Successfully uploaded {initial_image.filename}"}

import logging
from io import BytesIO
from PIL import Image
from typing import Optional

from torch import Generator

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from app.ml.StableDiffusion import text2imgPipe
from app.ml.StableDiffusion import img2imgPipe
from app.ml.StableDiffusion import inpaintingPipe
from app.ml.StableDiffusion import upscalerPipe
from app.ml.dis_bg_removal import inference as dis_inference
# Helpers
from app.utils import helpers

# Data Models
from app.utils.data_models import TextToImageGenerationParameters

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def health_check():
    return {"health_check": "healthy"}


@app.post("/text-to-image")
def generate_text_to_image(parameters: TextToImageGenerationParameters):
    logging.info(parameters)

    generator = Generator("cpu").manual_seed(parameters.seed)

    image_list = []
    for i in range(parameters.num_images_per_prompt):
        generated_image = text2imgPipe(
            prompt=parameters.prompt,
            height=parameters.height,
            width=parameters.height,
            num_inference_steps=parameters.num_inference_steps,
            guidance_scale=parameters.guidance_scale,
            negative_prompt=parameters.negative_prompt,
            generator=generator
        ).images[0]

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    return image_list


@app.post("/image-to-image")
async def generate_image_to_image(initial_image: UploadFile = File(...),
                                  prompt: str = Form(...),
                                  strength: float = Form(...),
                                  num_inference_steps: int = Form(...),
                                  guidance_scale: float = Form(...),
                                  negative_prompt: Optional[str] = Form(""),
                                  num_images_per_prompt: int = Form(...),
                                  seed: int = Form(...)):
    logging.info({"prompt": prompt})

    image_contents = initial_image.file.read()
    init_img = Image.open(BytesIO(image_contents)).convert("RGB")

    generator = Generator("cpu").manual_seed(seed)

    image_list = []
    for i in range(num_images_per_prompt):
        generated_image = img2imgPipe(prompt=prompt,
                                      image=init_img,
                                      strength=strength,
                                      num_inference_steps=num_inference_steps,
                                      guidance_scale=guidance_scale,
                                      negative_prompt=negative_prompt,
                                      generator=generator
                                      ).images[0]

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    return image_list


@app.post("/create-image-mask")
async def create_image_mask(image: UploadFile = File(...)):
    image_contents = image.file.read()
    image_bytes = BytesIO(image_contents)

    results = dis_inference(image_bytes)
    image_mask = results[1]

    image_string = helpers.get_image_string(image_mask)

    logging.info("mask generated")
    return image_string


@app.post("/image-inpainting")
async def generate_image_inpainting(initial_image: UploadFile = File(...),
                                    mask_image: UploadFile = File(...),
                                    prompt: str = Form(...),
                                    num_inference_steps: int = Form(...),
                                    guidance_scale: float = Form(...),
                                    negative_prompt: Optional[str] = Form(""),
                                    num_images_per_prompt: int = Form(...),
                                    seed: int = Form(...)):
    logging.info({"prompt": prompt})

    init_image_contents = initial_image.file.read()
    init_img = Image.open(BytesIO(init_image_contents)).convert("RGB")

    mask_image_contents = mask_image.file.read()
    mask_img = Image.open(BytesIO(mask_image_contents)).convert("RGB")

    initial_image_width, initial_image_height = init_img.size

    generator = Generator("cpu").manual_seed(seed)

    image_list = []
    for i in range(num_images_per_prompt):
        generated_image = inpaintingPipe(
            image=init_img,
            mask_image=mask_img,
            prompt=prompt,
            height=initial_image_height,
            width=initial_image_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    return image_list


@app.post("/super-resolution")
async def generate_super_resolution(initial_image: UploadFile = File(...),
                                    prompt: str = Form(...),
                                    num_inference_steps: int = Form(...),
                                    guidance_scale: float = Form(...),
                                    negative_prompt: Optional[str] = Form(""),
                                    num_images_per_prompt: int = Form(...),
                                    seed: int = Form(...)):
    logging.info({"prompt": prompt})

    image_contents = initial_image.file.read()

    low_res_img = Image.open(BytesIO(image_contents)).convert("RGB")

    initial_image_width, initial_image_height = low_res_img.size

    low_res_img = low_res_img.resize((128, 128))

    generator = Generator("cpu").manual_seed(seed)

    image_list = []
    for i in range(num_images_per_prompt):
        generated_image = upscalerPipe(prompt=prompt,
                                       image=low_res_img,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale,
                                       negative_prompt=negative_prompt,
                                       generator=generator
                                       ).images[0]

        generated_image = generated_image.resize((initial_image_width, initial_image_height))

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    return image_list

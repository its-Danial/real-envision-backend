from io import BytesIO
from PIL import Image
from typing import Optional

import torch

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from app.ml.StableDiffusion import text2imgPipe, img2imgPipe
from app.ml.dis_bg_removal import inference as create_image_mask
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
    print(parameters)

    generator = torch.Generator("cpu").manual_seed(parameters.seed)

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

        generated_image.save(f'test_image{i}.png')

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    print(image_list)

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
    print({"prompt": prompt,
           "strength": strength,
           "num_inference_steps": num_inference_steps,
           "guidance_scale": guidance_scale,
           "negative_prompt": negative_prompt,
           "num_images_per_prompt": num_images_per_prompt,
           "seed": seed})

    contents = initial_image.file.read()
    init_img = Image.open(BytesIO(contents)).convert("RGB")
    # init_img = init_img.resize((768, 512))

    # init_img.show()

    generator = torch.Generator("cpu").manual_seed(seed)

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

        generated_image.save(f'test_image{i}.png')

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    print(image_list)
    return image_list


@app.get("/image-inpainting")
async def generate_image_inpainting():
    result = create_image_mask("app/robot.png")
    result[1].show()

    return {"Running": "healthy"}

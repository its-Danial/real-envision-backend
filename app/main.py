from io import BytesIO
from PIL import Image
from typing import Optional

from torch import Generator

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from app.ml.StableDiffusion import text2imgPipe, img2imgPipe, inpaintingPipe
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
    print(parameters)

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

    image_contents = initial_image.file.read()
    init_img = Image.open(BytesIO(image_contents)).convert("RGB")
    # init_img = init_img.resize((768, 512))

    # init_img.show()

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

        generated_image.save(f'test_image{i}.png')

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    print(image_list)
    return image_list


@app.post("/create-image-mask")
async def create_image_mask(image: UploadFile = File(...)):
    image_contents = image.file.read()
    image_bytes = BytesIO(image_contents)

    results = dis_inference(image_bytes)

    image_mask = results[1]

    image_string = helpers.get_image_string(image_mask)
    print(image_string)
    return image_string


@app.post("/image-inpainting")
async def generate_image_inpainting(initial_image: UploadFile = File(...),
                                    mask_image: UploadFile = File(...),
                                    prompt: str = Form(...),
                                    height: int = Form(...),
                                    width: int = Form(...),
                                    num_inference_steps: int = Form(...),
                                    guidance_scale: float = Form(...),
                                    negative_prompt: Optional[str] = Form(""),
                                    num_images_per_prompt: int = Form(...),
                                    seed: int = Form(...)):

    # TODO: mask_image can be a string when it is generated instead of uploaded, introduce condition what type it is.
    init_image_contents = initial_image.file.read()
    init_img = Image.open(BytesIO(init_image_contents)).convert("RGB")

    mask_image_contents = mask_image.file.read()
    mask_img = Image.open(BytesIO(mask_image_contents)).convert("RGB")

    generator = Generator("cpu").manual_seed(seed)

    image_list = []
    for i in range(num_images_per_prompt):
        generated_image = inpaintingPipe(
            image=init_img,
            mask_image=mask_img,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]

        generated_image.save(f'test_image{i}.png')

        image_string = helpers.get_image_string(generated_image)

        image_list.append(image_string)

    print(image_list)

    return image_list

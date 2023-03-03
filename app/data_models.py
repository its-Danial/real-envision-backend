import typing

from fastapi import File, UploadFile
from pydantic import BaseModel


class TextToImageGenerationParameters(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 8.5
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    seed: int = 0




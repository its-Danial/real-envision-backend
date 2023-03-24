from io import BytesIO
import base64

import torch


def get_device_name():
    device: str
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_image_string(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_string = base64.b64encode(buffer.getvalue())
    return image_string

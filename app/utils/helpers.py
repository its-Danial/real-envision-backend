from io import BytesIO
import base64


def get_image_string(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_string = base64.b64encode(buffer.getvalue())
    return image_string

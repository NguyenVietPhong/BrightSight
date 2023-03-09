import torch
# import torchvision.transforms as transforms
# import onnxruntime
import numpy as np
from config import Config
# import onnxruntime as ort
from PIL import Image
def load_image(image_file, transform=None):
    image = Image.open(image_file)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image)
    image = np.array(image, dtype=np.float32)
    return image

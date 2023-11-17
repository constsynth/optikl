from PIL import Image
from optikl.model.srgan import generator
from optikl.model.common import resolve_single
import numpy as np
import os
import requests
import tarfile
from optikl.model.tools import Fundamental
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if not os.path.exists("./optikl/models/weights/srgan"):
    url = 'https://martin-krasser.de/sisr/weights-srgan.tar.gz'
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path="./optikl/models")
    print('Model ESRGAN has downloaded succesfully!')

def crop_center(image: Image.Image,
                frac: float = None) -> Image.Image:
    frac = frac
    left = image.size[0] * ((1 - frac) / 2)
    upper = image.size[1] * ((1 - frac) / 2)
    right = image.size[0] - ((1 - frac) / 2) * image.size[0]
    bottom = image.size[1] - ((1 - frac) / 2) * image.size[1]
    cropped_img = image.crop((left, upper, right, bottom))
    cropped_img = cropped_img.resize((256, 256))
    cropped_img.save('./sr_not_applied.jpg')
    return cropped_img

def super_resolution(image: Image.Image) -> Image.Image:
    model_sr_gan = generator(num_filters=64, num_res_blocks=16)
    model_sr_gan.load_weights('./cspackage/models/weights/srgan/gan_generator.h5')
    image = resolve_single(model_sr_gan, image)
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image.save('./sr_is_applied.jpg')
    return image

def represent(path_to_image: str = None, to_vector: bool = False, to_tensor: bool = True):
    method = Fundamental().to(device)
    return method.encode(path_to_image=path_to_image, to_vector=to_vector,to_tensor=to_tensor)

def image_similarity(im1_path: str = None, im2_path: str = None):
    method = Fundamental().to(device)
    return method.distance(im1_path=im1_path, im2_path=im2_path)
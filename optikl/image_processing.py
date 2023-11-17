from PIL import Image
from optikl.model.srgan import generator
from optikl.model.common import resolve_single
import numpy as np
import os
import requests
import tarfile
from optikl.model.tools import Fundamental
import torch
import cv2
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def crop_center(image_path: str = None,
                frac: float = None,
                resize: bool = False,
                resize_shape: tuple = (512, 512)) -> Image.Image:
    print(device)
    image = Image.open(image_path)
    frac = frac
    left = image.size[0] * ((1 - frac) / 2)
    upper = image.size[1] * ((1 - frac) / 2)
    right = image.size[0] - ((1 - frac) / 2) * image.size[0]
    bottom = image.size[1] - ((1 - frac) / 2) * image.size[1]
    cropped_img = image.crop((left, upper, right, bottom))
    if resize:
        cropped_img = cropped_img.resize(resize_shape)
    cropped_img.save(f'./cropped_{int(frac*100)}%.jpg')
    return cropped_img

def super_resolution(image_path: str = None,) -> Image.Image:
    print(device)
    image = Image.open(image_path)

    if not os.path.exists("./optikl/models/weights/srgan"):
        print('ESRGAN was not found, downloading...')
        url = 'https://martin-krasser.de/sisr/weights-srgan.tar.gz'
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path="./optikl/models")
        print('ESRGAN has downloaded succesfully!')

    model_sr_gan = generator(num_filters=64, num_res_blocks=16)
    model_sr_gan.load_weights('./optikl/models/weights/srgan/gan_generator.h5')
    image = resolve_single(model_sr_gan, image)
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save('./after_sr.jpg')
    return image

def represent(path_to_image: str = None, to_vector: bool = False, to_tensor: bool = True):
    print(device)
    method = Fundamental().to(device)
    return method.encode(path_to_image=path_to_image, to_vector=to_vector,to_tensor=to_tensor)

def image_similarity(im1_path: str = None, im2_path: str = None) -> float:
    print(device)
    method = Fundamental().to(device)
    return method.distance(im1_path=im1_path, im2_path=im2_path)

def find_face(img_path: str = None, show: bool = False, save: bool = True) -> dict:
    image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detection_model = cv2.CascadeClassifier(f"{os.path.abspath('haarcascade_frontalface_default.xml')}")
    face = face_detection_model.detectMultiScale(
        grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    faces = []
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
        x1 = face[0][0]
        y1 = face[0][1]
        x2 = face[0][0] + face[0][2]
        y2 = face[0][1] + face[0][3]
        square = ((x1, y1), (x2, y2))
        faces.append(square)
    image_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(image_to_show)
        plt.axis('off')
        plt.show()

    if save:
        cv2.imwrite('./result_face_recognition.jpg', image)

    return {'faces': faces}



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
import skimage
import mediapipe as mp

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

def super_resolution(image_path: str = None, save: bool = False) -> Image.Image:
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
    if save:
        image.save('./after_sr.jpg')
    return image

def represent(path_to_image: str = None, to_vector: bool = False, to_tensor: bool = True, type: str = 'linear'):
    # print(device)
    # method = Fundamental().to(device)
    # return method.encode(path_to_image=path_to_image, to_vector=to_vector,to_tensor=to_tensor, type=type)

    if not os.path.exists("./optikl/models/weights/mobilenet"):
        os.makedirs('./optikl/models/weights/mobilenet', exist_ok=True)
        print('MobileNetV3 was not found, downloading...')
        url = 'https://storage.googleapis.com/mediapipe-tasks/image_embedder/mobilenet_v3_small_075_224_embedder.tflite'
        response = requests.get(url, stream=True)
        open('./optikl/models/weights/mobilenet/mobilenet_v3_small_075_224_embedder.tflite', 'wb').write(response.content)
        print('MobileNetV3 has downloaded succesfully!')
    BaseOptions = mp.tasks.BaseOptions
    ImageEmbedder = mp.tasks.vision.ImageEmbedder
    ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path='./optikl/models/weights/mobilenet/mobilenet_v3_small_075_224_embedder.tflite'),
        quantize=True,
        running_mode=VisionRunningMode.IMAGE)

    with ImageEmbedder.create_from_options(options) as embedder:
        mp_image = mp.Image.create_from_file(path_to_image)
        embedding_result = embedder.embed(mp_image)
    return embedding_result.embeddings[0]


def image_similarity(im1_path: str = None, im2_path: str = None, type_of_encoding: str = 'linear', size = (512, 512), use_features: bool = False) -> float:
    ImageEmbedder = mp.tasks.vision.ImageEmbedder
    print(device)
    if use_features:
        im1 = represent(im1_path)
        im2 = represent(im2_path)
        similarity = ImageEmbedder.cosine_similarity(im1, im2)
    else:
        im1 =cv2.resize(cv2.imread(im1_path), size)
        im2 = cv2.resize(cv2.imread(im2_path), size)
        similarity = skimage.metrics.structural_similarity(im1, im2, channel_axis=-1)
    # method = Fundamental().to(device)
    # return method.distance(im1_path=im1_path, im2_path=im2_path, type_of_encoding=type_of_encoding)
    return similarity


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
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
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



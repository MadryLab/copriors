import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(2),
    ])

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])


ADDITIONAL_TRANSFORMS = {
    'GRAYSCALE': lambda: transforms.Grayscale(3),
    'NONE': lambda: None,
    'CANNY': lambda: transform_canny_edge, 
    'SOBEL': lambda: transform_sobel_edge,
}


# TODO: both of these are pretty inefficient...
def transform_canny_edge(img):
    img = Image.fromarray(cv2.bilateralFilter(np.array(img),5,75,75))
    gray_scale = transforms.Grayscale(1)
    image = gray_scale(img)
    edges = cv2.Canny(np.array(image), 100, 200)
    out = np.stack([edges, edges, edges], axis=-1)
    to_pil = transforms.ToPILImage()
    out = to_pil(out)
    return out

def transform_sobel_edge(img):
    curr_size = img.size[0]
    resize_up = transforms.Resize(max(curr_size, 128), 3)
    resize_down = transforms.Resize(curr_size, 3)
    rgb = np.array(resize_up(img))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb = cv2.GaussianBlur(rgb, (5,5), 5)
    sobelx = cv2.Sobel(rgb,cv2.CV_64F,1,0,ksize=3)
    imgx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(rgb,cv2.CV_64F,0,1,ksize=3)
    imgy = cv2.convertScaleAbs(sobely)
    tot = np.sqrt(np.square(sobelx) + np.square(sobely))
    imgtot = cv2.convertScaleAbs(tot)
    img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))
    resized = resize_down(img)
    return resized
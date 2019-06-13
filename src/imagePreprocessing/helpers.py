import cv2
import os
from collections import Counter

import matplotlib.pyplot as plt


# crop to square
from src.imagePreprocessing.tickBarsDetector import ImageResizerFactory


def crop_image(image):
    y_size, x_size, _ = image.shape
    dest_size = min(y_size, x_size)
    margin_x = x_size - dest_size
    margin_y = y_size - dest_size

    print('{}, {}'.format(margin_x, margin_y))

    image = image[int(margin_y/2):y_size - int(margin_y/2), int(margin_x/2):x_size - int(margin_x/2)]

    return image


image_resizer = ImageResizerFactory().columbia_images()

path = ''
files_list = os.listdir(path)  # returns list

for file in files_list:
    image = cv2.imread(path + file)
    image = crop_image(image)

    image_resizer.save_image(image, path + file)


import os
import xml.etree.cElementTree as ET
import re

import cv2


# moves images to dirs according to its tirads tag in relevant .xml
def separate_to_tirads_classes(path):
    files_list = os.listdir(path)  # returns list

    for file in files_list:
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.xml':
            tree = ET.parse(path + file)
            tirads = tree.findall('tirads')[0].text
            print(tirads)
            if tirads is None:
                pass
            else:
                for f in files_list:
                    matcher = re.compile(r'\b{}_'.format(filename))
                    if matcher.match(f):
                        print('     {}'.format(f))
                        new_path = '{}{}/{}'.format(path, tirads, f)
                        os.rename(path + f, new_path)


# resize image to given size
def resize_image(image, size):
    print(image.shape)
    return cv2.resize(image, (size, size))


# proceeds data augmentation by rotating images from -max_angle to max_angle
def rotate(image, max_angle):
    y_size, x_size, _ = image.shape
    angle_list = range(-max_angle, max_angle)

    print(angle_list)

    matrix_list = [cv2.getRotationMatrix2D((x_size / 2, y_size / 2), angle, 1) for angle in angle_list]
    images = [cv2.warpAffine(image.copy(), M, (x_size, y_size)) for M in matrix_list]

    return images


# crops image from black padding after rotation
def crop_image_after_rotation(image):
    y_size, x_size, _ = image.shape
    dest_size = min(y_size, x_size)

    # TODO automatically calculated add margin
    margin_x = x_size - dest_size + 70
    margin_y = y_size - dest_size + 70

    print('{}, {}'.format(margin_x, margin_y))

    return image[int(margin_y / 2):y_size - int(margin_y / 2), int(margin_x / 2):x_size - int(margin_x / 2)]


def read_image(path):
    return cv2.imread(path)


def save_image(image, path):
    cv2.imwrite(path, image)

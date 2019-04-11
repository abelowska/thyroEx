import numpy as np
import cv2
from functools import reduce


filename = "../../data/3.jpg"
image = cv2.imread(filename)
y_size, x_size, _ = image.shape

# kernel (y,x)
kernel = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# thresholding image
indexes = np.where((image >= [240, 240, 240]).all(axis=2))
image[indexes] = [0, 0, 0]

# restoring with similar neighbourhood; displayed as (y,x)
zipped_indexes = np.array(list(zip(indexes[0], indexes[1])))

for pixel in zipped_indexes:
    y, x = pixel

    neighbours = [(y + a, x + b) for a, b in kernel]
    neighbours_values = [image[yy, xx] for yy, xx in neighbours if 0 <= yy < y_size and 0 <= xx < x_size]

    # neighbour_values are uint_8 type so sum() causes overflow
    _sum = int(0)
    for item in neighbours_values:
        _sum += item[0]

    average_color = _sum/len(neighbours_values)
    image[y, x] = np.array([average_color] * 3)


# TODO add automatic cropping roi of image. Check annotations removing
cv2.imshow("Image", image)
cv2.waitKey(0)

# For second database. Needed grayscale - what a stupid USG machine. Annotations remover does not work good

import numpy as np
import cv2

filename = "../../data/6.jpg"
image = cv2.imread(filename)
y_size, x_size, _ = image.shape
sig_diff = 60
cut_off_threshold = 175

cv2.imshow("Image", image)
cv2.waitKey(0)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# cv2.imshow("Image", image)
# cv2.waitKey(0)


# kernel (y,x)
kernel = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# wiekszy kernel na wyczernianie jeszcze wiekszego obszaru doodola artefaktow

# thresholding image
indexes = np.where((image >= cut_off_threshold))
print(indexes)

dupa = np.where((image >= 0))
zipped_dupa = np.array(list(zip(dupa[0], dupa[1])))


# restoring with similar neighbourhood; displayed as (y,x)
zipped_indexes = np.array(list(zip(indexes[0], indexes[1])))
print(zipped_indexes)

pixels_with_bad_neighbours = []


for i in range(1):
    for pixel in zipped_indexes:
        y, x = pixel

        neighbours = [(y + a, x + b) for a, b in kernel if 0 <= y+a < y_size and 0 <= x + a < x_size]
        neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if 0 <= yy < y_size and 0 <= xx < x_size]

        filtered_neighbours = [item for item in neighbours_values if int(image[y, x]) - int(item[1]) >= sig_diff]

# 4 for blue one and 3 for standard images
        if len(filtered_neighbours) >= 3:
            image[y, x] = 0
            pixels_with_bad_neighbours.append(pixel)
            pixels_with_bad_neighbours.extend(neighbours)

for item in pixels_with_bad_neighbours:
        image[item[0], item[1]] = 255

# cv2.imshow("Image", image)
# cv2.waitKey(0)


for i in range(10):
    for pixel in pixels_with_bad_neighbours:
        y, x = pixel

        neighbours = [(y + a, x + b) for a, b in kernel]
        neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if
                             0 <= yy < y_size and 0 <= xx < x_size]

        # neighbour_values are array of uint_8 type so sum() causes overflow
        _sum = int(0)
        for value in neighbours_values:
            color = value[1]
            _sum += color

        average_color = int(_sum / len(neighbours_values))
        image[y, x] = average_color


# TODO add automatic cropping roi of image. Check annotations removing.

cv2.imshow("Image", image)
cv2.waitKey(0)

import numpy as np
import cv2

filename = "../../data/4.jpg"
image = cv2.imread(filename)
y_size, x_size, _ = image.shape
sig_diff = 100
cut_off_threshold = 235

# kernel (y,x)
kernel = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# thresholding image
indexes = np.where((image >= np.array([cut_off_threshold] * 3)).all(axis=2))
print(indexes)

# restoring with similar neighbourhood; displayed as (y,x)
zipped_indexes = np.array(list(zip(indexes[0], indexes[1])))

for i in range(5):
    for pixel in zipped_indexes:
        y, x = pixel

        neighbours = [(y + a, x + b) for a, b in kernel]
        neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if 0 <= yy < y_size and 0 <= xx < x_size]

        filtered_neighbours = [item for item in neighbours_values if int(image[y, x][0]) - int(item[1][0]) >= sig_diff]

        if len(filtered_neighbours) >= 4:
            image[y, x] = [0, 0, 0]

cv2.imshow("Image", image)
cv2.waitKey(0)

for i in range(5):

    for pixel in zipped_indexes:
        y, x = pixel

        neighbours = [(y + a, x + b) for a, b in kernel]
        neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if
                             0 <= yy < y_size and 0 <= xx < x_size]

        # neighbour_values are array of uint_8 type so sum() causes overflow
        _sum = int(0)
        for value in neighbours_values:
            _sum += value[1][0]

        average_color = _sum / len(neighbours_values)
        image[y, x] = np.array([average_color] * 3)


# TODO add automatic cropping roi of image. Check annotations removing.

cv2.imshow("Image", image)
cv2.waitKey(0)

import numpy as np
import cv2
import itertools
from collections import Counter

filename = "../../data/2.jpg"
image = cv2.imread(filename)
margin = 60
margin_y = 20
y = 0
x = 0

y_size, x_size, _ = image.shape

roi = image[y:margin_y, x + margin:x_size - margin]

# threshold the image
(_, thresh) = cv2.threshold(roi, 245, 255, cv2.THRESH_BINARY)

white_points = np.where(thresh == 255)
white_y_coordinates = white_points[0]
white_x_coordinates = white_points[1]
unique_y_coordinates = np.unique(white_y_coordinates)
unique_x_coordinates = np.unique(white_x_coordinates)

print(unique_x_coordinates)

pairs = list(itertools.product(unique_x_coordinates, repeat=2))
print(pairs)

diffs = [abs(x - y) for x, y in pairs]
print(diffs)

# for filtering data
threshold = 20

filtered_diffs = [x for x in diffs if x > threshold]

print(filtered_diffs)

diffs_dict = Counter(filtered_diffs)
print(diffs_dict)

bar_tick = diffs_dict.most_common(1)[0][0]
print(bar_tick)

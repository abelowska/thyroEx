import numpy as np
import cv2
import itertools
from collections import Counter
import matplotlib.pyplot as plt

# default resolution of USG image
DEFAULT_SIZE = 50

filename = "../../data/7.jpg"
image = cv2.imread(filename)
margin = 300
margin_y = 20
y = 0
x = 0

y_size, x_size, _ = image.shape

# for second database
image = image[y+120: y_size, x:x_size]

roi = image[y:margin_y, x + margin:x_size - margin].copy()

# threshold the image
(_, thresh) = cv2.threshold(roi, 230, 255, cv2.THRESH_BINARY)

# cv2.imshow("Image", thresh)
# cv2.waitKey(0)

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

plt.bar(diffs_dict.keys(), diffs_dict.values())
# plt.show()

bar_tick = diffs_dict.most_common(2)[1][0]
print(bar_tick)

if bar_tick != DEFAULT_SIZE:
    print("in resizing")
    scale = DEFAULT_SIZE/bar_tick
    image = cv2.resize(image, None, fx=scale, fy=scale)

cv2.imwrite("../../data/resized.jpg", image)
# cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


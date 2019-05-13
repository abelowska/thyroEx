import numpy as np
import cv2
import itertools
from collections import Counter

# default resolution of USG image
DEFAULT_SIZE = 100


class ImageResizer:
    def __init__(self, margin_x=60, margin_y=20):
        self.margin_x = margin_x
        self.margin_y = margin_y

    @staticmethod
    def read_image(path):
        return cv2.imread(path)

    # returns roi
    def crop_ticks_bar_area(self, image):
        y_size, x_size, _ = image.shape
        return image[0:self.margin_y, self.margin_x:x_size - self.margin_x].copy()

    @staticmethod
    def threshold_image(roi):
        (_, thresh) = cv2.threshold(roi, 245, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def find_white_points(thresh):
        white_points = np.where(thresh == 255)
        white_y_coordinates = white_points[0]
        white_x_coordinates = white_points[1]
        unique_y_coordinates = np.unique(white_y_coordinates)
        unique_x_coordinates = np.unique(white_x_coordinates)

        print(unique_x_coordinates)
        return unique_x_coordinates, unique_y_coordinates

    @staticmethod
    def calculate_tick(white_points_coordinates):
        x_coordinates = white_points_coordinates[0]
        pairs = list(itertools.product(x_coordinates, repeat=2))
        print(pairs)

        diffs = [abs(x - y) for x, y in pairs]
        print(diffs)

        # for filtering nearest points
        threshold = 20

        filtered_diffs = [x for x in diffs if x > threshold]
        print(filtered_diffs)

        diffs_dict = Counter(filtered_diffs)
        print(diffs_dict)

        bar_tick = diffs_dict.most_common(1)[0][0]
        print(bar_tick)

        return bar_tick

    @staticmethod
    def resize(bar_tick, image):
        if bar_tick != DEFAULT_SIZE:
            print("in resizing")
            scale = DEFAULT_SIZE / bar_tick
            image = cv2.resize(image, None, fx=scale, fy=scale)
            return image

    @staticmethod
    def save_image(image):
        cv2.imwrite("../../data/resized.jpg", image)


image_resizer = ImageResizer()
my_image = image_resizer.read_image("../../data/4.jpg")
roi = image_resizer.crop_ticks_bar_area(my_image.copy())
thresh_image = image_resizer.threshold_image(roi)
coordinates = image_resizer.find_white_points(thresh_image)
tick = image_resizer.calculate_tick(coordinates)

my_image = image_resizer.resize(tick, my_image)
image_resizer.save_image(my_image)

cv2.destroyAllWindows()

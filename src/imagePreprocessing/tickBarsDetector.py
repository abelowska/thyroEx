import itertools
from collections import Counter

import cv2
import numpy as np
import piexif


class ImageResizerFactory:
    @staticmethod
    def columbia_images():
        return ImageResizer(margin_x=60, margin_y=20, x=60, y=0, cut_off_threshold=245, special_ticks=0)

    @staticmethod
    def french_images():
        return ImageResizer(margin_x=300, margin_y=25, x=300, y=120, cut_off_threshold=230, special_ticks=1)


class ImageResizer:
    def __init__(self, margin_x, margin_y, x, y, cut_off_threshold, special_ticks):
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.x = x
        self.y = y
        self.cut_off_threshold = cut_off_threshold
        self.special_ticks = special_ticks

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
    def repair_list(diffs):
        repaired_diffs = []

        for item in diffs:
            if item in repaired_diffs:
                repaired_diffs.append(item)
            elif item+1 in repaired_diffs:
                repaired_diffs.append(item+1)
            elif item-1 in repaired_diffs:
                repaired_diffs.append(item-1)
            else:
                repaired_diffs.append(item)

        return repaired_diffs


    @staticmethod
    def resize(bar_tick, default_tick, image):
        # if bar_tick != default_tick:
        print("in resizing")
        scale = default_tick / bar_tick
        image = cv2.resize(image, None, fx=scale, fy=scale)
        return image

     # returns roi
    def crop_ticks_bar_area(self, image):
        y_size, x_size, _ = image.shape
        image = image[self.y:self.y + self.margin_y, self.x:x_size - self.margin_x].copy()
        return image

    def threshold_image(self, roi):
        (_, thresh) = cv2.threshold(roi, self.cut_off_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def calculate_tick(self, white_points_coordinates):
        x_coordinates = white_points_coordinates[0]
        pairs = list(itertools.product(x_coordinates, repeat=2))
        print(pairs)

        diffs = [abs(x - y) for x, y in pairs]
        print(diffs)

        # for filtering nearest points
        threshold = 20

        filtered_diffs = [x for x in diffs if x > threshold]
        print(filtered_diffs)

        diffs = ImageResizer.repair_list(filtered_diffs)

        diffs_dict = Counter(diffs)
        print(diffs_dict)

        # plt.bar(diffs_dict.keys(), diffs_dict.values())
        # plt.show()

        bar_tick = diffs_dict.most_common(self.special_ticks + 1)[self.special_ticks][0]

        print(bar_tick)
        return bar_tick

    @staticmethod
    def save_bar_tick(path, bar_tick):
        zeroth_ifd = {piexif.ImageIFD.DefaultScale: (bar_tick, 1)}
        exif_dict = {"0th": zeroth_ifd, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)

    @staticmethod
    def read_tick(path):
        exif_dict = piexif.load(path)
        bar_tick = exif_dict["0th"][piexif.ImageIFD.DefaultScale][0]

        print(bar_tick)
        return bar_tick

    def find_tick(self, image):
        # cv2.imshow("Image", my_image)
        # cv2.waitKey(0)
        roi = self.crop_ticks_bar_area(image)
        thresh_image = self.threshold_image(roi)
        coordinates = self.find_white_points(thresh_image)
        return self.calculate_tick(coordinates)


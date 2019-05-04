# Annotations remover does not work well
# works for first database (needed different threshold = 235 and sig_diff=100)
# sig_diff = 60 for second database
# cut_off_threshold = 175 for second database
# sig_neighbours = 4 for blue ones from second database. 3 for first and standard from second


import numpy as np
import cv2

# TODO wiekszy kernel na wyczernianie jeszcze wiekszego obszaru doodola artefaktow
# TODO add automatic cropping roi of image. Check annotations removing for first database.


# kernel (y,x)
KERNEL = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


class AnnotationRemover:
    def __init__(self, cut_off_threshold=175, sig_diff=60, sig_neighbours=4, steps=10):
        self.cut_off_threshold = cut_off_threshold
        self.sig_diff = sig_diff
        self.sig_neighbours = sig_neighbours
        self.steps = steps

    @staticmethod
    def read_image(path="../../data/10.jpg"):
        filename = path
        image = cv2.imread(filename)
        return image

    # TODO add fit name of changed image

    @staticmethod
    def save_image(image):
        cv2.imwrite("../../data/change.jpg", image)

    def find_annotations_with_neighbourhood(self, image):
        y_size, x_size, _ = image.shape
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)

        # change to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # thresholding image
        indexes = np.where((image >= self.cut_off_threshold))
        zipped_indexes = np.array(list(zip(indexes[0], indexes[1])))

        # we need not only significant diff pixels but its neigh. also because of gray shadow around annotations
        pixels_with_bad_neighbours = []

        for i in range(1):
            for pixel in zipped_indexes:
                y, x = pixel

                # all neighbours
                neighbours = [(y + a, x + b) for a, b in KERNEL if 0 <= y + a < y_size and 0 <= x + a < x_size]
                # neighbours with indexes and values
                neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if
                                     0 <= yy < y_size and 0 <= xx < x_size]
                # only significant neighbours
                filtered_neighbours = [item for item in neighbours_values if
                                       int(image[y, x]) - int(item[1]) >= self.sig_diff]

                if len(filtered_neighbours) >= self.sig_neighbours:
                    image[y, x] = 0
                    pixels_with_bad_neighbours.append(pixel)
                    pixels_with_bad_neighbours.extend(neighbours)

        for item in pixels_with_bad_neighbours:
            image[item[0], item[1]] = 255

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        return image, pixels_with_bad_neighbours

    def restore_gaps(self, image, pixels_with_bad_neighbours):
        #  for first database steps = 5 (but 10 is also ok)
        y_size, x_size = image.shape

        for i in range(self.steps):
            for pixel in pixels_with_bad_neighbours:
                y, x = pixel

                neighbours = [(y + a, x + b) for a, b in KERNEL]
                neighbours_values = [((yy, xx), image[yy, xx]) for yy, xx in neighbours if
                                     0 <= yy < y_size and 0 <= xx < x_size]

                # neighbour_values are array of uint_8 type so sum() causes overflow
                _sum = int(0)
                for value in neighbours_values:
                    color = value[1]
                    _sum += color

                average_color = int(_sum / len(neighbours_values))
                image[y, x] = average_color

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        return image


annotation_remover = AnnotationRemover()
my_image = annotation_remover.read_image()
my_image, pixels = annotation_remover.find_annotations_with_neighbourhood(image=my_image)
my_image = annotation_remover.restore_gaps(image=my_image, pixels_with_bad_neighbours=pixels)

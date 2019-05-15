import numpy as np
import cv2

# TODO add automatic cropping roi of image.


# kernel (y,x)
KERNEL = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
EDGE_LOWER_THRESHOLD = 30
EDGE_UPPER_THRESHOLD = 90


class AnnotationRemoverCreator:
    @staticmethod
    def blue_annotations():
        return AnnotationRemover(cut_off_threshold=175, sig_diff=60, sig_neighbours=4, steps=10)

    @staticmethod
    def columbia_annotations():
        return AnnotationRemover(cut_off_threshold=235, sig_diff=100, sig_neighbours=3, steps=10)

    @staticmethod
    def french_annotations():
        return AnnotationRemover(cut_off_threshold=175, sig_diff=60, sig_neighbours=3, steps=10)


class AnnotationRemover:
    def __init__(self, cut_off_threshold, sig_diff, sig_neighbours, steps):
        self.cut_off_threshold = cut_off_threshold
        self.sig_diff = sig_diff
        self.sig_neighbours = sig_neighbours
        self.steps = steps

    @staticmethod
    def read_image(path):
        return cv2.imread(path)

    # TODO add fit name of changed image

    @staticmethod
    def save_image(image):
        cv2.imwrite("../../data/change.jpg", image)

    # then voting for x and y coordinates
    @staticmethod
    def crop_image(image):
        y_size, x_size, _ = image.shape
        tmp_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), np.uint8)
        tmp_image = cv2.erode(tmp_image, kernel, iterations=2)
        kernel = np.ones((4, 4), np.uint8)
        tmp_image = cv2.dilate(tmp_image, kernel, iterations=4)

        # cv2.imshow("Image", tmp_image)
        # cv2.waitKey(0)

        indexes = np.where((tmp_image <= 20))
        zipped_indexes = np.array(list(zip(indexes[0], indexes[1])))

        for pixel in zipped_indexes:
            y, x = pixel

            neighbours = [(y + a, x + b) for a, b in KERNEL if 0 <= y + a < y_size and 0 <= x + a < x_size]
            neighbours_values = [((yy, xx), tmp_image[yy, xx]) for yy, xx in neighbours if
                                 0 <= yy < y_size and 0 <= xx < x_size]
            filtered_neighbours = [item for item in neighbours_values if int(item[1]) < 1]

            if len(filtered_neighbours) >= 3:
                tmp_image[y, x] = 0

        _, thresh = cv2.threshold(tmp_image.copy(), 0, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(thresh, EDGE_LOWER_THRESHOLD, EDGE_UPPER_THRESHOLD)
        cv2.imshow('Edges', edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # getting the biggest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        for contour in contours:
            # approximate the contour
            peri = cv2.arcLength(curve=contour, closed=True)
            approx = cv2.approxPolyDP(curve=contour, epsilon=0.01 * peri, closed=True)
            cv2.drawContours(image, [contour], -1, 125, 5)

        cv2.imshow("Image", image)
        cv2.waitKey(0)

        return None

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

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        return image


annotation_remover = AnnotationRemoverCreator.columbia_annotations()
my_image = annotation_remover.read_image(path="../../data/9.jpg")
my_image = annotation_remover.crop_image(my_image)
my_image, pixels = annotation_remover.find_annotations_with_neighbourhood(image=my_image)
my_image = annotation_remover.restore_gaps(image=my_image, pixels_with_bad_neighbours=pixels)

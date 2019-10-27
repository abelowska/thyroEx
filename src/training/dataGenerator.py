import random

import os
import cv2
from imutils import paths
from keras.utils import to_categorical
from keras_preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split


class DataGenerator:

    def load_data(self, path):
        print("[INFO] loading images...")

        image_paths = sorted(list(paths.list_images(path)))[:2500]
        print(image_paths)

        random.seed(42)
        random.shuffle(image_paths)

        (data, labels) = self.separate_classes(image_paths)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        print(len(data))
        print(len(labels))

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          labels, test_size=0.25, random_state=42)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=2)
        testY = to_categorical(testY, num_classes=2)

        return (trainX, trainY), (testX, testY)

    @staticmethod
    def separate_classes(image_paths):
        data = []
        labels = []

        for image_path in image_paths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(image_path)
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = image_path.split(os.path.sep)[-2]
            label = 1 if label == "malignant" else 0
            labels.append(label)

        return data, labels


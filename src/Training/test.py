import os
import random

import numpy as np
import matplotlib.pyplot as plt

import cv2
from keras.engine import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator, img_to_array

from src.googLeNet.google import create_model


train_bening_path = '/Users/Anna/work/train/bening/'
train_malignant_path = '/Users/Anna/work/train/malignant//'
test_bening_path = '/Users/Anna/work/test/bening/'
test_malignant_path = '/Users/Anna/work/test/malignant/'

imagePaths = os.listdir(train_bening_path)[:50]  # returns list
imagePaths2 = os.listdir(train_malignant_path)[:50]  # returns list
imagePaths3 = os.listdir(test_bening_path)[:50] # returns list
imagePaths4 = os.listdir(test_malignant_path)[:50]  # returns list
data = []
labels = []

val_data = []
val_labels = []
##################
# model = create_googlenet()
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')


################################
x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

# Create a Keras Model - Functional API
model = Model(input=img_input, output=[x])
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(train_bening_path + imagePath)
    image = img_to_array(image)
    data.append(image)
    labels.append(0)

for imagePath in imagePaths2:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(train_malignant_path + imagePath)
    image = img_to_array(image)
    data.append(image)
    labels.append(1)

for imagePath in imagePaths3:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(test_bening_path + imagePath)
    image = img_to_array(image)
    val_data.append(image)
    val_labels.append(0)

for imagePath in imagePaths4:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(test_malignant_path + imagePath)
    image = img_to_array(image)
    val_data.append(image)
    val_labels.append(1)


def create_image_labels_lists(path_1, path_2):
    files_list_1 = os.listdir(path_1)[:50]  # returns list
    random.shuffle(files_list_1)

    image_list_1 = [img_to_array(cv2.imread(path_1 + file)) for file in files_list_1]
    labels_list_1 = np.repeat(0, len(image_list_1))

    files_list_2 = os.listdir(path_2)[:50] # returns list
    random.shuffle(files_list_2)

    image_list_2 = [img_to_array(cv2.imread(path_2 + file)) for file in files_list_2]
    labels_list_2 = np.repeat(1, len(image_list_2))

    x = np.concatenate((image_list_1, image_list_2), axis=0)
    y = np.concatenate((labels_list_1, labels_list_2), axis=0)

    return x, y


# x_train, y_train = create_image_labels_lists(train_bening_path, train_malignant_path)
# x_test, y_test = create_image_labels_lists(test_bening_path, train_bening_path)
#
# x_train = x_train / 255.0
# x_test = x_test / 255.0


x_train = np.array(data, dtype="float") / 255.0
y_train = np.array(labels)

x_test = np.array(val_data, dtype="float") / 255.0
y_test = np.array(val_labels)

# x_train = np.swapaxes(x_train, 1, 3)
# x_test = np.swapaxes(x_test, 1, 3)


print('{} {}'.format(x_train.shape, y_train.shape))
# print('{} {}'.format(x_train[0], x_train[1]))

num_classes = 2

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

print('{} {}'.format(x_train.shape, y_train.shape))


datagen = ImageDataGenerator()
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
#
# fits the model on batches with real-time data augmentation:
# H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32)
#                     # validation_data=(x_test, y_test), epochs=1)

# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = 1
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on Santa/Not Santa")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
#
# # here's a more "manual" example
#
for e in range(1):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        print(x_batch.shape)
        print(y_batch.shape)
        loss = model.train_on_batch(x_batch, y_batch) # note the three outputs
        # batches += 1
        # if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            # break

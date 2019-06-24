import matplotlib
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator

from src.googLeNet.googlenet import create_googlenet

matplotlib.use("Agg")

from keras.engine import Model
from src.training.dataGenerator import DataGenerator
from src.googLeNet.google import create_model
import matplotlib.pyplot as plt
import numpy as np

# initialize the number of epochs to train for, initial learning rate and batch size
EPOCHS = 15
INIT_LR = 1e-3
BS = 32
PATH = '/Users/Anna/work/'

dg = DataGenerator()
(x_train, y_train), (x_test, y_test) = dg.load_data(PATH)

x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)
# y_train = np.swapaxes(y_train, 1, 3)
# y_test = np.swapaxes(y_test, 1, 3)

print('{} {}'.format(x_train.shape, y_train.shape))

###############
datagen = ImageDataGenerator()
datagen.fit(x_train)

# Create model
# x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()
model = create_googlenet()

print("[INFO] compiling model...")
# Create a Keras Model - Functional API
# model = Model(input=img_input, output=[x])
# model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# train the network
print("[INFO] training network...")
# H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BS),
#                         steps_per_epoch=len(x_train) / 32,
#                         validation_data=(x_test, y_test), epochs=EPOCHS)

for e in range(EPOCHS):
    batches = 0
    for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=BS):
        loss = model.train_on_batch(X_batch, [Y_batch, Y_batch, Y_batch]) # note the three outputs
        batches += 1
        if batches >= len(x_train) / BS:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break


# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = 1
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("training Loss and Accuracy on Santa/Not Santa")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")

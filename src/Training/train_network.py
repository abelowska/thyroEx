import matplotlib
from keras_preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")

from keras.engine import Model
from src.Training.dataGenerator import DataGenerator
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


print('{} {}'.format(x_train.shape, y_train.shape))


###############
datagen = ImageDataGenerator()
datagen.fit(x_train)

# Create model
x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

print("[INFO] compiling model...")
# Create a Keras Model - Functional API
model = Model(input=img_input, output=[x])
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# train the network
print("[INFO] training network...")
H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BS),
                    steps_per_epoch=len(x_train) / 32,
                    validation_data=(x_test, y_test), epochs=EPOCHS)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = 1
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

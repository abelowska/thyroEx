import matplotlib
import tf as tf
from keras import Input
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
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


print('{} {}'.format(x_train.shape, y_train.shape))

###############
datagen = ImageDataGenerator()
datagen.fit(x_train)

# Create model
input, model = create_googlenet("../googLeNet/googlenet_weights.h5")
print("[INFO] compiling model...")

# for index, layer in enumerate(model.layers):
#     print("{} {}".format(str(index), str(layer.name)))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# Plugging into old net
loss1_classifier = Dense(2, name='loss1/classifier', kernel_regularizer=l2(0.0002))(model.layers[100].output)
loss2_classifier = Dense(2, name='loss2/classifier', kernel_regularizer=l2(0.0002))(model.layers[101].output)
loss3_classifier = Dense(2, name='loss3/classifier', kernel_regularizer=l2(0.0002))(model.layers[102].output)

loss1_classifier_act = Activation('softmax')(loss1_classifier)
loss2_classifier_act = Activation('softmax')(loss2_classifier)
loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

googlenet = Model(inputs=model.layers[0].output,
                  outputs=[loss3_classifier_act])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
googlenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
googlenet.summary()
model = googlenet
# train the network
print("[INFO] training network...")


# y_test = np.swapaxes([y_test]*3, 0, 1)
# y_train = np.swapaxes([y_train]*3, 0, 1)

# ValueError: `x` (images tensor) and `y` (labels) should have the same length. Found: x.shape = (975, 3, 224, 224), y.shape = (3, 975, 2)
# y_train = np.swapaxes(np.array([y_train] * 3), 1, 0)
# y_test = np.swapaxes(np.array([y_test] * 3), 1, 0)
# y_train = np.swapaxes(np.array([y_train]*3), 0, 1)
# y_test = tuplize(y_test, 3)
print("[INFO] reshaped the arrays")
flow = datagen.flow(x_train, y_train, batch_size=BS)
validation_flow = datagen.flow(x_test, y_test)
print("[INFO] created datagen flow")
steps = len(x_train) / 32
print("[INFO] steps {}".format(steps))
H = model.fit_generator(flow,
                        steps_per_epoch=steps,
                        validation_data=validation_flow, epochs=EPOCHS)

print(H)
print("INFO: calculating accuracy")
loss, acc = model.evaluate_generator(validation_flow, steps=steps)
print("\n%s: %.2f%%" % (model.metrics_names[1], acc * 100))

# for e in range(EPOCHS):
#     batches = 0
#     print("INFO: Starting epoch {}".format(e))
#     for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=BS):
#         print("INFO: Training on next batch in {}".format(e))
#         loss = model.train_on_batch(X_batch, [Y_batch, Y_batch, Y_batch]) # note the three outputs
#         print("INFO: Loss {}".format(loss))
#         batches += 1
#
#         if batches >= len(x_train) / BS:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break

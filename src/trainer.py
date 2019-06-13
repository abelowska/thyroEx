from keras.optimizers import SGD

from src.dataGenerator import DataGenerator
from src.googLeNet.googlenet import create_googlenet
from keras import backend as K
K.set_image_data_format('channels_first')


class CNNTrainer:
    def __init__(self, training_set, test_set):
        self.training_set = training_set
        self.test_set = test_set

    def train(self, classifier):
        # training_set = DataGenerator.get_train_set()
        # test_set = DataGenerator.get_test_set()

        classifier.fit_generator(
            self.training_set,
            steps_per_epoch=2000,
            epochs=3,
            validation_data=self.test_set,
            validation_steps=200)


model = create_googlenet()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


training_set = DataGenerator.get_train_set()
test_set = DataGenerator.get_test_set()

trainer = CNNTrainer(training_set, test_set)
trainer.train(model)

from keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    @staticmethod
    def get_train_set():
        train_data_gen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_data_gen.flow_from_directory(
            '/home/mikegpl/Desktop/IM/thyroid/work/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')

        return training_set

    @staticmethod
    def get_test_set():

        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        test_set = test_data_gen.flow_from_directory(
            '/home/mikegpl/Desktop/IM/thyroid/work/test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')

        return test_set

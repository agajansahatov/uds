from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


class ModelBuilder:
    def __init__(self, image_height, image_width, image_channels):
        self.input_shape = (image_height, image_width, image_channels)

    def build_model1(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPool2D())
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dropout(0.20))
        model.add(Dense(16))
        model.add(Dense(1))
        model.summary()
        return model

    def build_model2(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.summary()
        return model

    def build_model3(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(MaxPool2D((2, 2), padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(MaxPool2D((2, 2), padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(MaxPool2D((2, 2), padding='same'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64, activation='elu'))
        model.add(Dense(16, activation='elu'))
        model.add(Dense(1))
        model.summary()
        return model


if __name__ == '__main__':
    model_builder = ModelBuilder(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    model_builder.build_model1()
    model_builder.build_model2()
    model_builder.build_model3()

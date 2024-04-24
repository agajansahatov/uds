from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D

class ModelBuilder:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model1(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        # Add more layers as needed
        return model

    def build_model2(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        # Add layers for model 2
        return model

    def build_model3(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        # Add layers for model 3
        return model

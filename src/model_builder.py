from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D


class ModelBuilder:
    def __init__(self, input_shape=(66, 200, 3)):
        self.input_shape = input_shape

    def build_model1(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=self.input_shape))
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        # Normalize the values to between -1 and 1, all data, 127.5-1,
        # improve model efficiency (image data is between 0-255),
        # without affecting the single precision of the image
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPool2D())
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPool2D())
        model.add(Flatten())
        # The number of output nodes (the output size dimension is 32)
        model.add(Dense(32))
        model.add(Dropout(0.20))
        model.add(Dense(16))
        model.add(Dense(1))
        # This method will display the number of parameters for each layer of the model,
        # as well as the total number of parameters of the entire model
        # and the number of trainable parameters.
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
        # This method will display the number of parameters of each layer,
        # as well as the number of parameters of the entire model
        # and the number of parameters that can be trained
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


# Usage
if __name__=='__main__':
    model_builder = ModelBuilder()
    model_builder.build_model1()
    model_builder.build_model2()
    model_builder.build_model3()

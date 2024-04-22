import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

class ModelBuilder:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, model_type):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        model.add(tf.keras.layers.Lambda(lambda x: x / 127.5 - 1))
        if model_type == 1:
            self._build_model1(model)
        elif model_type == 2:
            self._build_model2(model)
        elif model_type == 3:
            self._build_model3(model)
        return model

    def _build_model1(self, model):
        model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.20))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

    def _build_model2(self, model):
        model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation='elu'))
        model.add(tf.keras.layers.Dense(50, activation='elu'))
        model.add(tf.keras.layers.Dense(10, activation='elu'))
        model.add(tf.keras.layers.Dense(1))

    def _build_model3(self, model):
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.MaxPool2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.MaxPool2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
        model.add(tf.keras.layers.MaxPool2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='elu'))
        model.add(tf.keras.layers.Dense(16, activation='elu'))
        model.add(tf.keras.layers.Dense(1))
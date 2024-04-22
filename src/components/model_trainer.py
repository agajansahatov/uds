import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import ModelCheckpoint, EarlyStopping, TensorBoard
from src.components.image_preprocessor import ImagePreprocessor
from src.components.model_builder import ModelBuilder

class ModelTrainer:
    def __init__(self, data_path, test_ratio=0.1, batch_size=100, batch_num=200, epochs=50):
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.epochs = epochs

    def load_data(self):
        data_csv = pd.read_csv(self.data_path + 'driving_log.csv', names=['center', 'left', 'right', 'steering', '_', '__', '___'])
        X = data_csv[['center', 'left', 'right']].values
        Y = data_csv['steering'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, random_state=0)
        return X_train, X_test, Y_train, Y_test

    def train(self):
        X_train, X_test, Y_train, Y_test = self.load_data()
        model = ModelBuilder(input_shape=(66, 200, 3)).build_model(2)  # Choosing model type 2
        data_processor = ImagePreprocessor(image_height=66, image_width=200)
        checkpoint = ModelCheckpoint('model_{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, verbose=1, mode='auto')
        tensor_board = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=1, write_images=0)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
        model.fit(
            data_processor.batch_generator(self.data_path, self.batch_size, X_train, Y_train, True),
            steps_per_epoch=self.batch_num,
            epochs=self.epochs,
            verbose=1,
            validation_data=data_processor.batch_generator(self.data_path, self.batch_size, X_test, Y_test, False),
            validation_steps=1,
            max_queue_size=1,
            callbacks=[checkpoint, stopping, tensor_board]
        )
        model.save('agajan_lake_model.h5')

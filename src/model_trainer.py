import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src.preprocessor import Preprocessor
from src.model_builder import ModelBuilder


class ModelTrainer:
    def __init__(self, data_path='data-lake/', test_ration=100, batch_size=100, batch_num=200, epoch=50):
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.epoch = epoch
        self.test_ration = test_ration
        self.preprocessor = Preprocessor()

    def load_data(self):
        # Load data from CSV
        # names: Annotates the data
        data_csv = pd.read_csv(self.data_path + 'driving_log.csv',
                               names=['center', 'left', 'right', 'steering', '_', '__', '___'])
        x = data_csv[['center', 'left', 'right']].values
        y = data_csv['steering'].values
        # Split into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_ration, random_state=0)
        return x_train, x_test, y_train, y_test

    def batch_generator(self, data_path, batch_size, x_data, y_data, train_flag):
        image_height, image_width, image_channels = self.preprocessor.get_image_info()
        image_container = np.empty([batch_size, image_height, image_width, image_channels])
        steer_container = np.empty(batch_size)
        while True:
            ii = 0
            for index in np.random.permutation(x_data.shape[0]):
                center, left, right = data_path + x_data[index]
                steering_angle = y_data[index]
                if train_flag and np.random.rand() < 0.4:
                    image, steering_angle = self.preprocessor.preprocess_image(center, left, right, steering_angle)
                else:
                    image = cv2.imread(center)
                image_container[ii] = self.preprocessor.normalize_image(image)
                steer_container[ii] = steering_angle
                ii += 1
                if ii == batch_size:
                    break
            yield image_container, steer_container

    def train_model(self):
        x_train, x_test, y_train, y_test = self.load_data()
        input_shape = (self.preprocessor.get_image_info())

        model_builder = ModelBuilder(input_shape)
        model = model_builder.build_model2()

        checkpoint = ModelCheckpoint(
            'agajan_lake_model_{epoch:03d}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=200,
            verbose=1,
            mode='auto'
        )
        tensor_board = TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=1,
            write_images=0
        )

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
        model.fit(
            self.batch_generator(self.data_path, self.batch_size, x_train, y_train, True),
            steps_per_epoch=self.batch_num,
            epochs=self.epoch,
            verbose=1,
            validation_data=self.batch_generator(self.data_path, self.batch_size, x_test, y_test, False),
            validation_steps=1,
            max_queue_size=1,
            callbacks=[checkpoint, stopping, tensor_board]
        )
        model.save('agajan_lake_model.h5')

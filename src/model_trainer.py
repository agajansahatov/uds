import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src.preprocessor import Preprocessor
from src.model_builder import ModelBuilder
from src.config import *


class ModelTrainer:
    def __init__(self, data_path='data-lake/', test_ration=0.1, batch_size=100, batch_num=200, epoch=50):
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.epoch = epoch
        self.test_ration = test_ration
        self.preprocessor = Preprocessor(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        self.model_builder = ModelBuilder(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    def load_data(self):
        data_csv = pd.read_csv(self.data_path + 'driving_log.csv', names=['center', 'left', 'right', 'steering',
                                                                     '_', '__', '___'])  # names:为数据加标注
        # print(data_csv)
        X = data_csv[['center', 'left', 'right']].values  # 有监督学习，将数据分为“输入”+“标签”
        # print(X)
        Y = data_csv['steering'].values  # 标签：期望的输出值
        # print(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ration, random_state=0)
        # X:数据（输入）；Y:标签（输出）分成训练集和测试集
        # print(X_train,X_test,Y_train,Y_test)
        return X_train, X_test, Y_train, Y_test

    def generate_batch(self, data_path, batch_size, X_data, Y_data, train_flag):
        image_container = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])  # 定义容器，盛放数据
        steer_container = np.empty(batch_size)
        while True:
            ii = 0
            for index in np.random.permutation(X_data.shape[0]):  # range(),np.random.choice()不同
                center, left, right = data_path + X_data[index]
                steering_angle = Y_data[index]
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

    def train_model(self, save_path='models/', file_name='agajan_model'):
        X_train, X_test, Y_train, Y_test = self.load_data()
        model = self.model_builder.build_model2()
        checkpoint = ModelCheckpoint(
            (save_path + file_name + '_{epoch:03d}.h5'),  # “xinglina"改为自己名字全拼
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
            self.generate_batch(self.data_path, self.batch_size, X_train, Y_train, True),
            steps_per_epoch=self.batch_num,
            epochs=self.epoch,
            verbose=1,
            validation_data=self.generate_batch(self.data_path, self.batch_size, X_test, Y_test, False),
            validation_steps=1,
            max_queue_size=1,
            callbacks=[checkpoint, stopping, tensor_board]
        )
        model.save(f'{save_path}{file_name}.h5')


if __name__ == "__main__":
    model_trainer = ModelTrainer('../data-lake/')
    model_trainer.train_model('../models/', 'agajan_lake_model')

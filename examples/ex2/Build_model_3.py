'''
太原理工大学现代科技学院（毕业）实习（实训）课程
项目一 基于深度学习的 Udacity 无人驾驶系统
组号：
学号：              姓名：
'''
#完成工作：搭建卷积神经网络模型
# 1、导入第三方库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from Preprocessing_2 import image_height, image_width, image_channels

# 2、搭建卷积神经网络模型
Input_size = (image_height, image_width, image_channels)
def build_model1():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
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


def build_model2():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
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


def build_model3():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))
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


# 3、设置主函数
if __name__ == '__main__':
    build_model1()
    build_model2()
    build_model3()
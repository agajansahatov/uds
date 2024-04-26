import cv2
import numpy as np
from src.config import *


class Preprocessor:
    def __init__(self, image_height, image_width, image_channels):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels

    def choose_image(self, center, left, right, steering_angle):
        choice = np.random.choice(3)
        # print(choice)
        if choice == 0:
            image_name = center
            bias = 0.0
        if choice == 1:
            image_name = left
            bias = 0.2
        if choice == 2:
            image_name = right
            bias = -0.2
        image = cv2.imread(image_name)
        steering_angle = steering_angle + bias
        # cv2.imshow('image_choose',image)
        # cv2.waitKey(0)
        return image, steering_angle

    def translate_image(self, image, steering_angle):
        range_X, range_Y = 100, 10
        tran_X = int(range_X * (np.random.rand() - 0.5))
        tran_Y = int(range_Y * (np.random.rand() - 0.5))
        tran_m = np.float32([[1, 0, tran_X], [0, 1, tran_Y]])
        image = cv2.warpAffine(image, tran_m, (image.shape[1], image.shape[0]))
        steering_angle = steering_angle + tran_X * 0.002
        # cv2.imshow('image_translate',image)
        # cv2.waitKey(0)
        return image, steering_angle

    def flip_image(self, image, steering_angle):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
            # cv2.imshow('image_flip', image)
            # cv2.waitKey(0)
        return image, steering_angle

    def normalize_image(self, image):
        image = image[60:-25, :, :]
        image = cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # cv2.imshow('image_normalized',image)
        # cv2.waitKey(0)
        return image

    def preprocess_image(self, center, left, right, steering_angle):
        image, steering_angle = self.choose_image(center, left, right, steering_angle)
        image, steering_angle = self.translate_image(image, steering_angle)
        image, steering_angle = self.flip_image(image, steering_angle)
        # image=image_normalized(image)
        return image, steering_angle


if __name__ == '__main__':
    preprocessor = Preprocessor(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    preprocessed_image, preprocessed_steering_angle = preprocessor.preprocess_image("../test/center.jpg",
                                                                                    '../test/left.jpg',
                                                                                    '../test/right.jpg',
                                                                                    STEERING_ANGLE)
    normalized_image = preprocessor.normalize_image(preprocessed_image)
    cv2.imshow('image_data', normalized_image)
    cv2.waitKey(0)
    print(preprocessed_steering_angle)
    cv2.destroyAllWindows()

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, image_height=66, image_width=200, image_channels=3):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels

    def choose_image(self, center, left, right, steering_angle):
        choice = np.random.choice(3)
        if choice == 0:
            image_name = center
            bias = 0.0
        elif choice == 1:
            image_name = left
            bias = 0.2
        else:
            image_name = right
            bias = -0.2
        image = cv2.imread(image_name)
        steering_angle += bias
        return image, steering_angle

    def flip_image(self, image, steering_angle):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def translate_image(self, image, steering_angle):
        range_x, range_y = 100, 10
        tran_x = int(range_x * (np.random.rand() - 0.5))
        tran_y = int(range_y * (np.random.rand() - 0.5))
        tran_m = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
        image = cv2.warpAffine(image, tran_m, (image.shape[1], image.shape[0]))
        steering_angle += tran_x ** 0.02
        return image, steering_angle

    def normalize_image(self, image):
        image = image[60:-25, :, :]
        image = cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return image

    def process_image(self, center, left, right, steering_angle):
        image, steering_angle = self.choose_image(center, left, right, steering_angle)  # 调用图像选择
        image, steering_angle = self.flip_image(image, steering_angle)
        image, steering_angle = self.translate_image(image, steering_angle)
        # image = normalize_image(image)
        return image, steering_angle


if __name__ == '__main__':
    preprocessor = Preprocessor()
    image, steering_angle = preprocessor.process_image("../test/center.jpg", '../test/left.jpg', '../test/right.jpg', 0.0)
    image = preprocessor.normalize_image(image)
    cv2.imshow('image_data', image)
    cv2.waitKey(0)
    print(steering_angle)
    cv2.destroyAllWindows()

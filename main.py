import cv2
from src.components.image_preprocessor import ImagePreprocessor

if __name__ == '__main__':
    center = 'test/center.jpg'
    left = 'test/left.jpg'
    right = 'test/right.jpg'
    steering_angle = 0.0

    preprocessor = ImagePreprocessor()
    image, steering_angle = preprocessor.choose_image(center, left, right, steering_angle)
    image, steering_angle = preprocessor.flip_image(image, steering_angle)
    image, steering_angle = preprocessor.translate_image(image, steering_angle)
    image = preprocessor.normalize_image(image)

    print(steering_angle)
    cv2.imshow('image_data', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
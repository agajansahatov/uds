import socketio
import eventlet.wsgi
from flask import Flask
import base64, cv2
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from preprocessor import ImagePreprocessor


class AutoDriver:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.image_preprocessor = ImagePreprocessor()

    def process_image(self, image_data):
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_preprocessor.normalize_image(image)
        return image

    def predict_steering_angle(self, image):
        steering_angle = self.model.predict(np.array([image]))
        return steering_angle

    def drive(self):
        # Code for establishing connection and controlling the car
        pass

import socketio
import eventlet.wsgi
from flask import Flask
import base64
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessor import Preprocessor
from src.pi_controller import PIController
import logging

logging.basicConfig(level=logging.INFO)


class AutoDriver:
    def __init__(self, model_path='models/agajan_lake_model.h5', speed=15, steering_angle=-0.02, throttle=0.3):
        self.model = load_model(model_path)
        self.preprocessor = Preprocessor()
        self.pi_controller = PIController(0.1, 0.002)
        self.speed = speed
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.sio = socketio.Server()

    def send_control(self, steering_angle, throttle):
        self.sio.emit('steer', data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        })

    def process_image(self, image_data):
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Image from Udacity Simulator', image)
        cv2.waitKey(1)
        return self.preprocessor.normalize_image(image)

    def handle_telemetry(self, sid, data):
        if data:
            speed = float(data['speed'])
            image = self.process_image(data['image'])
            steering_angle = float(self.model.predict(np.array([image])))
            throttle = self.pi_controller.updated(speed)
            self.send_control(steering_angle, throttle)
        else:
            self.sio.emit('manual', data={})

    def run(self):
        # Create network connection
        app = Flask(__name__)
        app = socketio.WSGIApp(self.sio, app)

        # Pass parameters to control car driving
        @self.sio.on('connect')
        def on_connect(sid, environ):
            logging.info('Successfully connected to the emulator！')

        @self.sio.on('telemetry')
        def on_telemetry(sid, data):
            self.handle_telemetry(sid, data)

        @self.sio.on('disconnect')
        def on_disconnect(sid):
            logging.info('Disconnected from emulator')

        # Start running
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == "__main__":
    auto_driver = AutoDriver('../models/agajan_lake_model.h5')
    auto_driver.run()

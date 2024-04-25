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


class AutoDriver:
    def __init__(self, model_path, speed=15, steering_angle=-0.02, throttle=0.3):
        self.model = load_model(model_path)
        self.pi_controller = PIController(0.1, 0.002)
        # Create network connection
        self.sio = socketio.Server()
        self.app = socketio.WSGIApp(self.sio, Flask(__name__))

    def send_control(self, steering_angle, throttle):
        self.sio.emit('steer', data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        })


    def run(self):
        # Pass parameters to control car driving
        @self.sio.on('connect')
        def on_connect(sid, environ):
            print('Successfully connected to the simulator！')

        @self.sio.on('telemetry')
        def on_telemetry(sid, data):
            if data:
                # print('收到信息',data)
                speed = float(data['speed'])
                # print('speed', speed)
                image = Image.open(BytesIO(base64.b64decode(data['image'])))
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('Image from Udacity Simulator', image)
                cv2.waitKey(1)
                # print(image)
                # throttle=1.0-steering_angle**2-(speed/set_speed)**2
                preprocessor = Preprocessor()
                image = preprocessor.normalize_image(image)
                steering_angle = float(self.model.predict(np.array([image])))
                throttle = self.pi_controller.updated(speed)
                self.send_control(steering_angle, throttle)
            else:
                self.sio.emit('manual', data={})

        @self.sio.on('disconnect')
        def on_disconnect(sid):
            print('Disconnected from simulator')

        # Start running
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)


if __name__ == "__main__":
    auto_driver = AutoDriver('../xinglina_lake_model2.h5')
    auto_driver.run()

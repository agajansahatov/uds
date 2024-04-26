'''
太原理工大学现代科技学院（毕业）实习（实训）课程
项目一 基于深度学习的 Udacity 无人驾驶系统
组号：
学号：              姓名：
'''
#完成工作：实现自动驾驶
#1、导入第三方库
#(1)互联网通信类
import socketio
import eventlet.wsgi
from flask import Flask
# #(2)图像处理类
import base64,cv2
from io import BytesIO
from PIL import Image
import numpy as np
#(3)模型相关类
from tensorflow.keras.models import load_model
from Preprocessing_2 import image_normalized
model=load_model('xinglina_lake_model2.h5')

#2、设置初始化变量
max_speed=15
steering_angle=-0.02
throttle=0.3
def send_control(steering_angle,throttle):
    sio.emit('steer',data={
        'steering_angle':steering_angle.__str__(),
        'throttle':throttle.__str__()
    })

#3、创建互联网连接
sio=socketio.Server()
app=Flask(__name__)
app=socketio.WSGIApp(sio,app)

#4、控制汽车运行
@sio.on('connect')
def on_connect(sid,environ):
    print('与模拟器连接成功')

@sio.on('telemetry')
def on_telemetry(sid,data):
    if data:
        #print(data)
        speed=float(data['speed'])
        #print(speed)
        image=Image.open(BytesIO(base64.b64decode(data['image'])))
        image=np.array(image)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow('Image from Udacity Simulator',image)
        cv2.waitKey(1)
        #print(image)

        image=image_normalized(image)
        steering_angle=float(model.predict(np.array([image])))


        throttle=1.0-steering_angle**2-(speed/max_speed)**2
        send_control(steering_angle,throttle)
    else:
        sio.emit('manual',data={})


@sio.on('disconnect')
def on_disconnect(sid):
    print('与模拟器断开')

#5、启动汽车运行
eventlet.wsgi.server(eventlet.listen(('',4567)),app)
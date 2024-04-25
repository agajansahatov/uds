'''太原理工大学计算机学院实习实训课程
基于深度学习的Udacity无人驾驶系统
组号：    学号：           姓名：
完成工作：1、数据扩充——解决数据量不够的问题
        2、丰富数据类型——让模型‘见多识广'
        3、数据归一化处理'''
#1、导入第三方库
import cv2
import numpy as np
#2、设置初始化变量
image_height,image_width,image_channels=66,200,3
#定义图像的长、宽，数据的通道数
center,left,right='./test/center.jpg','./test/left.jpg','./test/right.jpg'
#左、中、右三幅图像，对应于图片所在目录及名称
steering_angle=0.0
#3、选择图像
def iamge_choose(center,left,right,steering_angle):
    choice=np.random.choice(3)#在0,1,2中随机选取数字，随机选取图像
    if choice==0:
        image_name=center
        bias=0.0
    if choice==1:
        image_name=left
        bias=0.2
    if choice==2:
        image_name=right
        bias=-0.2
    image=cv2.imread(image_name)
    #cv2.imshow('image_choose',image)
    #cv2.waitKey(0)
    steering_angle=steering_angle+bias
    return image,steering_angle
#4、翻转图像
def image_flip(image,steering_angle): #定义翻转图像函数，输入变量：图像、转向角
    if np.random.rand()<0.5: #50%的概率翻转图像
        image=cv2.flip(image,1) #1为水平翻转；0为垂直翻转；-1为水平垂直同时翻转
        steering_angle=-steering_angle #偏向角取反
    #cv2.imshow('image_flip',image)
    #cv2.waitKey(0)
    return image,steering_angle
#5、平移图像
def image_translate(image,steering_angle):
    range_x,range_y=100,10 #定义移动的范围，水平范围100，垂直范围10
    tran_x=int(range_x*(np.random.rand()-0.5))
    tran_y = int(range_y * (np.random.rand() - 0.5))
    tran_m=np.float32([[1,0,tran_x],[0,1,tran_y]])
    image=cv2.warpAffine(image,tran_m,(image.shape[1],image.shape[0]))
    #调用仿射函数，先宽后高
    steering_angle=steering_angle+tran_x**0.02 #根据移动的距离进行方向调整
    #cv2.imshow('image_translate',image)
    #cv2.waitKey(0)
    return image,steering_angle

#6、归一化图像
def image_normalized(image):
    image=image[60:-25,:,:] #对图片进行裁剪，去除一些无关紧要的因素，例如：天空
    image=cv2.resize(image,(image_width,image_height),cv2.INTER_AREA)
    #对图像统一大小200x66，定义图像宽和高
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    #cv2.imshow('image_normalized',image)
    #cv2.waitKey(0)
    return image

#7、定义图像预处理函数
#for i in range(5):
def image_preprocessing(center,left,right,steering_angle):
    image,steering_angle=iamge_choose(center,left,right,steering_angle)#调用图像选择
    image, steering_angle =image_flip(image,steering_angle)
    image, steering_angle =image_translate(image, steering_angle)
    #image=image_normalized(image)
    return image,steering_angle
#8、设置主函数
if __name__=='__main__':
    image,steering_angle=image_preprocessing(center,left,right,steering_angle)
    image = image_normalized(image)
    cv2.imshow('image_data',image)
    cv2.waitKey(0)
    print(steering_angle)
    cv2.destroyAllWindows()
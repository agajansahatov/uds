'''
太原科技大学自动化专业（毕业）实习（实训）课程
项目二 基于深度学习的 Udacity 无人驾驶系统
组号：3组
学号：              姓名：
'''
#完成工作：1、数据的扩充；2、模拟真实情况，让模型实现“见多识广”；3、归一化数据
#1、导入第三方库
import cv2
import numpy as np
#2、设置初始化变量
image_height,image_width,image_channels=66,200,3
center,left,right='./test/center.jpg','./test/left.jpg','./test/right.jpg'
steering_angle=0.0
#3、选择图像
def image_choose(center,left,right,steering_angle):
    choice=np.random.choice(3)
    #print(choice)
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
    steering_angle=steering_angle+bias
    # cv2.imshow('image_choose',image)
    # cv2.waitKey(0)
    return image,steering_angle

#4、平移图像
def image_translate(image,steering_angle):
    range_X,range_Y=100,10
    tran_X=int(range_X*(np.random.rand()-0.5))
    tran_Y=int(range_Y*(np.random.rand()-0.5))
    tran_m=np.float32([[1,0,tran_X],[0,1,tran_Y]])
    image=cv2.warpAffine(image,tran_m,(image.shape[1],image.shape[0]))
    steering_angle=steering_angle+tran_X*0.002
    # cv2.imshow('image_translate',image)
    # cv2.waitKey(0)
    return image, steering_angle
#5、图像翻转
def image_flip(image,steering_angle):
    if np.random.rand()<0.5:
        image=cv2.flip(image,1)
        steering_angle=-steering_angle
        # cv2.imshow('image_flip', image)
        # cv2.waitKey(0)
    return image, steering_angle

#6、归一化图像
def image_normalized(image):
    image=image[60:-25,:,:]
    image=cv2.resize(image,(image_width,image_height),cv2.INTER_AREA)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    # cv2.imshow('image_normalized',image)
    # cv2.waitKey(0)
    return image


#7、图像处理
# for i in range(5):
def image_preprocessing(center,left, right, steering_angle):
    image,steering_angle=image_choose(center,left, right, steering_angle)
    image, steering_angle=image_translate(image,steering_angle)
    image, steering_angle = image_flip(image, steering_angle)
    # image=image_normalized(image)
    return image,steering_angle

#8、作为主函数运行
if __name__=='__main__':
    image, steering_angle=image_preprocessing(center,left, right, steering_angle)
    image=image_normalized(image)
    cv2.imshow('image_data',image)
    cv2.waitKey(0)
    print(steering_angle)
    cv2.destroyAllWindows()
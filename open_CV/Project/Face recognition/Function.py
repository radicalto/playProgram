import os

import numpy as np
from PIL import Image
import cv2
#人脸检测
def face_detect_demo(img):
    gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier("D:\\pythonProject\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml")
    face = face_detect.detectMultiScale(gary,1.1)
    # 可以就这两个参数，其他让他自适应 face_detect.detectMultiScale(gary,1.1,5,0,(100,100),(500,500))
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('result',img)

# 数据训练
def getImageAndLabels(path):
    i=0
    # 储存人脸数据
    facesSamples=[]
    # 储存姓名数据
    ids=[]
    # 储存图片信息
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv2.CascadeClassifier("D:\\pythonProject\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml")
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片 灰度化PIL有九种不同的模式 这里我们用L
        PIL_img = Image.open(imagePath).convert('L')
        # 将图片转化成数组，以黑白深浅
        img_numpy = np.array(PIL_img,'uint8')
        # 获取图片上的人脸特征
        img_numpy = cv2.imread(imagePath,0)
        faces = face_detector.detectMultiScale(img_numpy)
        # detectMultiScale它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用：
        # 获取每张图的人脸id和姓名
        # id = int(imagePath.split('/')[-1].split('.')[0])
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容照片
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
        print("id: %s,fs: %s"%(id,facesSamples[i]))
        i+=1
    return ids,facesSamples
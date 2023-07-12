import os

import cv2
cap = cv2.VideoCapture(0)           # 开启摄像头
f, frame = cap.read()               # 将摄像头中的一帧图片数据保存
cv2.imwrite('D:\\image.jpg', frame)
print(open('D:\\image.jpg', "rb").read())
os.remove('D:\\image.jpg')



# import cv2
# a=cv2.imread('D:\\pythonProject\\open_CV\\image\\girl_hiking.jpg')
# #retval,dst=cv2.threshold(src,thresh,maxval,type)
# retval,dst=cv2.threshold(a,127,255,cv2.THRESH_BINARY)#cv2.THRESH_BINARY 二进制阈值 大于阈值为255 还有其他很多模式
# print(retval)
# cv2.ellipse()
# cv2.namedWindow("threshold",0)
# cv2.resizeWindow("threshold",400,300)
# cv2.imshow("threshold",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
import numpy as np
# 读取图片 不同于pillow库，读取的是BGR且输出为数组类型
img = cv2.imread('D:\\pythonProject\\open_CV\\image\\1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.ellipse(gray,(256,256),(100,50),0,0,180,255,-1)
cv2.imshow("threshold",gray)
cv2.waitKey()
cv2.destroyAllWindows()
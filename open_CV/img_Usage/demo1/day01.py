import cv2
import numpy as np
# #500 x 250
# img1 = cv2.imread('D:\\pythonProject\\open_CV\\image\\animal\\animal1\\all.jpg')
# img2 = cv2.imread('D:\\pythonProject\\open_CV\\image\\animal\\animal2\\all.jpg')
# img3 = img2+img1
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img1)

import cv2 as cv

cap = cv.VideoCapture(0)
while True:
    # 获取一帧又一帧
    ret, frame = cap.read()
    cv.imshow("video", frame)

    # 方法一
    if cv.waitKey(1) == 27:         # 按下 Esc退出 (27是按键ESC对应的ASCII值)
        break
    # 方法二
    if cv.waitKey(1) == ord("e"):   # 按'e' 退出，也可以设置其他键
        break
while(1):
    if cv.getWindowProperty("video",0) == -1:
        break
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()

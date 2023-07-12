from Function import face_detect_demo
import cv2
cap = cv2.VideoCapture(0)


while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv2.waitKey(0):
        break

cv2.destroyAllWindows()
cap.release() # 释放摄像头
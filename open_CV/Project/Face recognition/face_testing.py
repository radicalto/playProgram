import cv2
from Function import face_detect_demo
#

img = cv2.imread("../../image/girl_hiking.jpg")

face_detect_demo(img)

while True:
    if ord('q') == cv2.waitKey(0):
        break

cv2.destroyAllWindows()

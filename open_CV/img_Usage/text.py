import cv2
import numpy as np
# 创建黑色图片
img = np.zeros((512,512,3),np.uint8)
# 绘制多边形
pts = np.array([[100,100],[100,200],[200,200],[180,150]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'polyline',(10,500), font, 4,255,2,cv2.LINE_AA)
cv2.imshow("drawFunc",img)
cv2.waitKey()# 这里可以理解为延长imshow显示时间
cv2.destroyAllWindows()# 关闭窗口
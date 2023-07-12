# 特征匹配＋全景拼接
import numpy as np
import cv2
from Project.Func import cv_show

# 读取拼接图片（注意图片左右的放置）
# 是对右边的图形做变换
img_right = cv2.imread('../../image/right.png')
img_left = cv2.imread('../../image/left.png')

img_right = cv2.resize(img_right, None, fx=1, fy=1.5)
# 保证两张图一样大
img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))

def sift_kp(image):
    # 初始化SIFT特征点检测器
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kimDraw = cv2.drawKeypoints(gray,kp,None)
    return kimDraw ,kp ,des

kimDraw_right,kp1,des1= sift_kp(img_right)
kimDraw_left,kp2,des2= sift_kp(img_left)

cv_show('kimd1',kimDraw_right)
cv_show('kimd2',kimDraw_left)

def good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good=[]
    matches = sorted(matches,key=lambda x:x[0].distance)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
goodMatch = good_match(des1,des2)
good_img = cv2.drawMatches(img_right,kp1,img_left,kp2,goodMatch[:10],None,flags=2)
cv_show('goodmatch',good_img)

def final_all(img_right,img_left):
    _,kp1,des1 = sift_kp(img_right)
    _,kp2,des2 = sift_kp(img_left)
    goodMatch = good_match(des1,des2)
    # 点的坐标 那来去最优的4向数据来计算3*3的矩阵H
    if len(goodMatch)>4:
        pstA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1,1,2)
        pstB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1,1,2)

        # 去最优的匹配项
        H,status = cv2.findHomography(pstA,pstB,cv2.RANSAC,4)
        # 透视变换，先弄右边  将图片右进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        cv_show('result_medium', result)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result

result = final_all(img_right,img_left)
cv_show('result',result)


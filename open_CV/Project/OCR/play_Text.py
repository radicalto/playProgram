from Project.Func import cv_show
import cv2
import numpy as np

# # 连接成线
# a = cv2.imread('../../image/1.jpg')
# b = cv2.resize(a,(400,600))
# cv2.line(b,[0,0],[400,600],(0,255,0),2)
# cv_show("line",b)

image = cv2.imread('../../image/OCRTest.jpg')
ratio = image.shape[0] / 500.0
#为什么是500.0 是因为我想他换H=500的衣服，看看比率
# 自定义图片缩放
def resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, cv2.INTER_AREA)
    return resized
#换上了品如的衣服：
orig=image.copy()  #后面还要copy
image = resize(orig, width=None, height=500)

# 预处理，该为灰度图，增加处理速度
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv_show('gray',gray)
# 高斯滤波除噪
Gauss_gray = cv2.GaussianBlur(gray,(5,5),0)
# canny边缘检测  点
canny_img = cv2.Canny(Gauss_gray.copy(),100,200)
# cv_show("canny_img",canny_img)
# 图像轮廓   线
contours = cv2.findContours(canny_img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
# 找最大的轮廓
max_list_contours = sorted(contours,key=cv2.contourArea,reverse=True)[:5]
# 筛选并进行轮廓近似，画矩形
screenCnt=[]
for cnt in max_list_contours:
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        screenCnt = approx
        break
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)  # 计算矩阵的每一行元素相加之和
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    # return the ordered coordinates
    return rect

rect = order_points(screenCnt.reshape(4, 2) * ratio)
(tl, tr, br, bl) = rect
print(tl,'\n', tr,'\n', br,'\n', bl)
cv2.line(orig,tuple(map(int, tl)),tuple(map(int, tr)),(0,255,0),2)
cv2.line(orig, tuple(map(int, tr)), tuple(map(int, br)),(0,255,0),2)
cv2.line(orig, tuple(map(int, br)),tuple(map(int, bl)),(0,255,0),2)
cv2.line(orig,tuple(map(int, bl)),tuple(map(int, tl)),(0,255,0),2)
cv_show("line",cv2.resize(orig,(800,500)))
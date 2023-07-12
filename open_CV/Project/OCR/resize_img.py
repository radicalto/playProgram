from Project.Func import *
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
print(screenCnt,np.diff(screenCnt.reshape(4,2), axis=1))
'''
[[245 146]
 [ 82 153]
 [103 423]
 [285 397]]

[391 235 526 682]

[[-99]
 [ 71]
 [320]
 [112]]
'''
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


def four_point_tramsform(image, pts):
    # 获取输入的坐标
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(tl, '\n', tr, '\n', br, '\n', bl)
    # 计算出输入的w和h的值  勾股定理  走起
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换之后的坐标位置
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype='float32')
    # 计算变换矩阵M  rect是轮廓四个点  dst是我们规定的四个点（利用W和H人为创造的）
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

warped = four_point_tramsform(orig, screenCnt.reshape(4, 2) * ratio)
# screenCnt是 Canny检测得到的  因为缩放 每个点的位置都要改
# 别忘了*比率 因为我们现在这个图orig是初始image的复制体

# cv_show('warped', warped)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
retval,ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)
# cv2.imwrite('scan.jpg', ref)

# 展示结果
# cv2.imshow('Original', resize(orig, height=650))
cv2.imshow('Scanned', resize(ref, height=650))
cv2.waitKey(0)

# 画轮廓


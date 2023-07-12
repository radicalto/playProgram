import argparse
import cv2
import imutils.contours
import numpy as np
from Project.Func import cv_show
'''
对于这类任务，先整理一下图像处理的思路。
我们输入的是一张答题卡的拍摄图片，而我们要处理的是这张答题卡的内容，需要用到透视变换将答题卡的内容单独拿出来；
提取答题卡中填涂区域的轮廓，并进行二值化处理，利用掩模与二值化后的答题卡进行对比处理。
'''
# 先定义了一个字典，存放了正确答案
ANSWER_KEY = {0: 2, 1: 4, 2: 0, 3: 3, 4: 1}

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='../../image/Answer_card.png',
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[0]

cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)

# 对轮廓按面积大小排序
fit_cnt = []
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            fit_cnt = approx
            break
cv2.rectangle(image, (112, 191), (433, 604), (0, 255, 0), 2)
cv_show('draw', image)
print('cnts: ', len(cnts))

# 透视变换
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)  # 计算和
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)  # 计算差 y-x
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    # (tl, bl, br, tr) = rect
    (bl, br, tr, tl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype='float32')
    H = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight))
    return warped


warped = four_point_transform(gray, fit_cnt.reshape(4, 2))
cv_show('warped', warped)

# 二值化处理
dst = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 轮廓检测 找到每个圆圈
contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print('contours:', len(contours))
draw_circle = cv2.drawContours(dst.copy(), contours, -1, (0, 0, 255), 2)
cv_show('draw_circle', draw_circle)

'''
接着绘制每一个填涂圆圈的掩模，由于填涂后的答题卡在二值图像中>0的像素点较多，而且掩模中的圆圈部分的像素值为255，
其余部分的像素值为0，将掩模与原图像进行“与”操作，得到每一个圆圈的“与”运算结果，判断该选项的圆圈是否被填涂了。
'''
# 获取答题选项轮廓
questionCnts = []
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
        print(len(questionCnts))
# 按从左到右在从上到下排列
# print('questionCnts:', len(questionCnts))
questionCnts = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]
# 初始化的得分为0
correct = 0
# 遍历不同题目
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = imutils.contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None
    # 制作mask
    for (j, cnt) in enumerate(cnts):
        mask = np.zeros(dst.shape, dtype='uint8')
        cv2.drawContours(mask, [cnt], -1, 255, 2)
        cv_show('mask', mask)
        mask = cv2.bitwise_and(dst, dst, mask=mask)
        # cv2.countNonZero函数的作用是统计非零像素点。
        # 也就是判断这个选项涂没涂上。
        total = cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    # 对比正确答案
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    cv2.drawContours(warped, [cnts[k]], -1, color, 3)
# 计算分数
score = (correct / 5) * 100
cv2.putText(warped,"{:.2f}%".format(score),(10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)

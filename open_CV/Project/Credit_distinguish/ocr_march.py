import imutils.contours
import numpy as np
from Project.Func import cv_show
import cv2
# 读取图像，并转化为灰度图
img = cv2.imread('../../image/number.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv_show('gray',gray)
# 二值化
two_value = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)[1]
# cv_show('two_value',two_value)# 轮廓检测
contours,hierarchy = cv2.findContours(two_value.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# RETR_EXTERNAL只要外轮廓

cv2.drawContours(img,contours,-1,(0,0,255),2)
cv_show('draw_contours',img)
print(np.array(contours,dtype=object).shape)
# print(type(contours),len(contours),contours[0])

# 遍历轮廓
refCnts = imutils.contours.sort_contours(contours, method="left-to-right")[0] #排序，从左到右，从上到下
print(type(refCnts))
digits = {}

for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = two_value[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每一个数字对应每一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，预处理
image = cv2.imread('../../image/credit.png')

image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray', gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1相当于用3*3的
                  ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# print((minVal, maxVal),(gradX == 2884).any() )
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))) # 对他做归一化操作
gradX = gradX.astype("uint8")
# cv_show('gradX1',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# cv_show('gradX2',gradX)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show('thresh', thresh)

# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
cv_show('thresh2',thresh)

# 计算轮廓
cnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# RETR_EXTERNAL只要外轮廓
cnt_img = image.copy()
cv2.drawContours(cnt_img,cnts,-1,(0,0,255),3)
cv_show('cnt_img',cnt_img)
# 遍历轮廓
locs = []
for (i, c) in enumerate(cnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / h
    if ar > 2.5 and ar < 4:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))
# 从左到右讲各块数字排起来
locs.sort(key=lambda x:x[0])
#相当于
'''
def f(x):
    return x[0]
locs.sort(key=f)'''
# 遍历每一块数字
Output = []
print('locs:',locs)
for (i, (x, y, w, h)) in enumerate(locs):
    groupOutput = []
    print(x,y,w,h)
    # group = cv2.rectangle(gray,(x - 5, y-5 ),(x+w+5,y + h + 5),(0,0,255),2)
    # 取每一块数字
    group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
    # 预处理
    group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    print(group)
    cv_show('group', group)
    # 取每一块里面的每个数字轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 轮廓从左到右排序
    digitCnts = imutils.contours.sort_contours(digitCnts,method="left-to-right")[0]
    # 取每一块里面的每个数字
    for item in digitCnts:
        (gx,gy,gw,gh) = cv2.boundingRect(item)
        roi = group[gy:gy + gh, gx:gx + gw]
        roi = cv2.resize(roi, (57, 88))
        cv_show('num',roi)
        # 匹配得分
        scores = []
        for (digit,digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            print(result,'\n',score)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    Output.append(groupOutput)


print("Credit Card #: {}".format("".join(str(Output))))
cv_show('finally',image)
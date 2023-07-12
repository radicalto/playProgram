# 特征匹配＋全景拼接
import numpy as np
import cv2
from Project.Func import cv_show
import PIL
# 读取拼接图片（注意图片左右的放置）
# 是对右边的图形做变换
img_right = cv2.imread('../../image/right.png')
img_left = cv2.imread('../../image/left.png')

img_right = cv2.resize(img_right, None, fx=1, fy=1.5)
# 保证两张图一样大
img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))


def sift_kp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 初始化sift特征点检测器
    sift = cv2.SIFT_create()
    # 检测特征点和描述符
    kp, des = sift.detectAndCompute(image, None)  # 返回关键点位置信息 keypoints,特征点向量descriptors
    kp_image = cv2.drawKeypoints(gray, kp, None)  # 绘制特征关键点
    return kp_image, kp, des


kpimg_right, kp1, des1 = sift_kp(img_right)
kpimg_left, kp2, des2 = sift_kp(img_left)
# 同时显示原图和关键点检测后的图
cv_show('img_left', np.hstack((img_left, kpimg_left)))
cv_show('img_right', np.hstack((img_right, kpimg_right)))


# 最佳匹配项
def get_good_match(des1, des2):
    # 创建匹配器 BF 蛮力匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    '''
    DMatch结构含有： 
    DMatch.distance：描述符之间的距离，越低越好。
    DMatch.queryIdx：主动匹配的描述符组中描述符的索引。
    DMatch.trainIdx：被匹配的描述符组中描述符的索引
    '''
    print(matches[0][1].distance)
    matches = sorted(matches, key=lambda x: x[0].distance)
    # 保存较好匹配项
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)# cv.drawMatchesKnn()把列表作为匹配项 要good.append([m])
    return good


good = get_good_match(des1, des2)



goodMatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, good[:10], None, flags=2)
# goodMatch_img2 = cv2.drawMatchesKnn(img_right,kp1,img_left,kp2,good[:10],None,flags=2)
cv_show('Keypoint Matches2', goodMatch_img)


# cv_show('Keypoint Matches4', goodMatch_img2)

# 全景拼接
def siftimg_rightlignment(img_right, img_left):
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        # kp[index].pt pt是元组 tuple(x,y) x=kp[index].pt[0] y=kp[index].pt[1]
        # DMatch.queryIdx：主动匹配的描述符组中描述符的索引。
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        # DMatch.trainIdx：被匹配的描述符组中描述符的索引
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

        #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H(retval)矩阵。H为3*3矩阵
        retval, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)
        print(retval, '\n', mask)
        # 将图片右进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img_right, retval, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        cv_show('result_medium', result)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result


# 把图片拼接成全景图
result = siftimg_rightlignment(img_right, img_left)
cv_show('result', result)

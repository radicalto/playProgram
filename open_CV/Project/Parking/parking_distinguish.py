import pickle
import cv2
import numpy as np
from Project.Func import cv_show

# import operator
# import pickle
# from keras.models import load_model

image = cv2.imread('../../image/P.jpg')
# cv_show('src', image)
# 掩模 去除背景
# lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
lower = np.array([120, 120, 120], dtype='uint8')
upper = np.array([255, 255, 255], dtype='uint8')
# 制作掩模
white_mask = cv2.inRange(image, lower, upper)
# 与操作（只显示掩膜部分的图像）实现去除背景
white_img = cv2.bitwise_and(image, image, mask=white_mask)
# cv_show('white_img',white_img)
# 转化为灰度图
gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
# 进行canny边缘检测
edge_image = cv2.Canny(gray, 50, 200)
# cv_show('edge_image', edge_image)
# 手动画点，制作mask
# 手动选择区域
row, col = image.shape[:2]
pt_1 = [col * 0.05, row * 0.90]
pt_2 = [col * 0.05, row * 0.70]
pt_3 = [col * 0.30, row * 0.55]
pt_4 = [col * 0.6, row * 0.15]
pt_5 = [col * 0.90, row * 0.15]
pt_6 = [col * 0.90, row * 0.90]

vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
point_img = edge_image.copy()
point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
for point in vertices[0]:
    cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
# cv_show('point_img', point_img)
# 填涂制作mask
mask = np.zeros_like(edge_image)
# mask2 = np.zeros(edge_image.shape, np.uint8)
# print(mask2.shape,mask.shape)
cv2.fillPoly(mask, vertices, 255)
cv_show('mask', mask)
# 过滤 只留下mask白色部分
ROI_image = cv2.bitwise_and(edge_image, mask)  # 并集
# cv_show('ROI_image',ROI_image)
lines = cv2.HoughLinesP(ROI_image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)
# print(lines.shape,lines[0])
'''
src：输入图像，必须8-bit的灰度图像
rho：生成极坐标时候的像素扫描步长
theta：生成极坐标时候的角度步长
threshold：阈值，只有获得足够交点的极坐标点才被看成是直线
lines：输出的极坐标来表示直线
minLineLength：最小直线长度，比这个短的线都会被忽略。
maxLineGap：最大间隔，如果小于此值，这两条直线 就被看成是一条直线。
'''
# 挑选出合适的直线并画出
ok_lines = []
line_image = image.copy()
print(line_image.shape)
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
        ok_lines.append((x1, y1, x2, y2))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
# print("No lines detected: ", len(ok_lines))
cv_show('line_image', line_image)
sort_lines_list = sorted(ok_lines, key=lambda x: (x[0], x[1]))
print('sort_lines_list', len(sort_lines_list))
# 将在同一列的直线放到字典中的一个键中
'''
判断两条直线是否是同一列的方法是：如果两条直线的左端点的横坐标x1相差小于20，
则认为这两条直线处于同一列当中；否则它们属于不同的列。'''
same_lines = {}
sum_line_num = 0
for i in range(len(sort_lines_list) - 1):
    distance = abs(sort_lines_list[i + 1][0] - sort_lines_list[i][0])
    if distance <= 15:
        if not sum_line_num in same_lines.keys():
            same_lines[sum_line_num] = []
        same_lines[sum_line_num].append(sort_lines_list[i])
        same_lines[sum_line_num].append(sort_lines_list[i + 1])
    else:
        sum_line_num += 1
print('same_lines', same_lines.keys())
# 得到矩形坐标
'''
因为提取出的直线中有一些是重复的，所以要先用set()函数将它们剔除掉，得到新的直线列表list2。
另外，一列中至少有5条直线，所以一列中没有5条直线的被认为是错误的列。
对list2中的直线进行排序，排序原则：直线左端点的纵坐标越大，直线越靠后。
从而可以得到一列中最前面和最后面的纵坐标，即矩形的上下两边界的纵坐标。
对于矩形左右两边界的横坐标，采用取此列中所有直线两端点横坐标平均值的方法获得。
'''
rect_dict = {}
index = 0
for key in same_lines:
    all_list = same_lines[key]
    diff_line_list = list(set(all_list))
    if len(diff_line_list) > 5:
        new_list = sorted(diff_line_list, key=lambda x: x[1])
        avg_y1 = new_list[0][1]
        avg_y2 = new_list[-1][1]
        avg_x1 = 0
        avg_x2 = 0
        for i in new_list:
            avg_x1 += i[0]
            avg_x2 += i[2]
        avg_x1 = avg_x1 / len(new_list)
        avg_x2 = avg_x2 / len(new_list)
        rect_dict[index] = (avg_x1, avg_y1, avg_x2, avg_y2)
        index += 1
rect_image = image.copy()
print(rect_dict.keys())
for key in rect_dict:
    (avg_x1, avg_y1, avg_x2, avg_y2) = rect_dict[key]
    topLeft = (int(avg_x1) - 10, int(avg_y1))
    bottom_right = (int(avg_x2) + 8, int(avg_y2))
    cv2.rectangle(rect_image, topLeft, bottom_right, (0, 255, 0), 2)
cv_show('rect', rect_image)
# 在每个列区域中画出横线
'''
第一列和最后一列每行只有一个停车位。其他列每行有两个停车位。
所以，不仅要在每一列中画出横线，还要在除第一列和最后一列的列区域中间画出竖线。
'''
# 每个区域画出横线
delineated = np.copy(image)
gap = 15.4  # 同一列中相邻停车位之间的纵向距离
spot_pos = {}
tot_spots = 0
# 微调
adj_y1 = {0: 0, 1: -10, 2: 15, 3: -11, 4: 28, 5: 30, 6: -30, 7: -25, 8: 5, 9: -15, 10: 45, 11: 0}
adj_y2 = {0: 10, 1: 10, 2: 15, 3: 0, 4: 0, 5: 15, 6: 15, 7: 10, 8: -16, 9: 15, 10: 15, 11: 30}

adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -12, 6: -15, 7: -15, 8: -13, 9: -12, 10: -10, 11: 2}
adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 10, 6: 10, 7: 10, 8: 7, 9: 10, 10: 10, 11: 4}

for key in rect_dict:
    tup = rect_dict[key]
    x1 = int(tup[0] + adj_x1[key])
    x2 = int(tup[2] + adj_x2[key])
    y1 = int(tup[1] + adj_y1[key])
    y2 = int(tup[3] + adj_y2[key])
    cv2.rectangle(delineated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv_show('delineated',delineated)
    num_splits = int(abs(y2 - y1) // gap)
    pt1 = (x1, y1)
    for i in range(num_splits + 1):
        y = int(y1 + i * gap)
        cv2.line(delineated, (x1, y), (x2, y), (0, 255, 0), 2)
    if key > 0 and key < len(rect_dict) - 1:
        x = int((x1 + x2) / 2)
        cv2.line(delineated, (x, y1), (x, y2), (0, 255, 0), 2)
    if key == 0 or key == len(rect_dict) - 1:
        tot_spots += num_splits + 1
    else:
        tot_spots += 2 * (num_splits + 1)
    if key == 0 or key == len(rect_dict) - 1:
        for i in range(num_splits + 1):
            cur_len = len(spot_pos)
            y = int(y1 + i * gap)
            spot_pos[(x1, y, x2, y + gap)] = cur_len + 1
    else:
        for i in range(num_splits + 1):
            cur_len = len(spot_pos)
            x = int((x1 + x2) / 2)
            y = int(y1 + i * gap)
            spot_pos[(x1, y, x, y + gap)] = cur_len + 1
            spot_pos[(x, y, x2, y + gap)] = cur_len + 2
print("total parking spaces: ", tot_spots, cur_len)
cv_show('delineated', delineated)

# 将得到的每个停车位的位置信息写入文件
with open('../../data/spot_dict.pickle', 'wb') as handle:
    pickle.dump(spot_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 将得到的每个停车位裁剪下来做成样本
for spot in spot_pos:
    (x1, y1, x2, y2) = spot
    (x1, y1, x2, y2) = (int(x1),int(y1),int(x2),int(y2))
    spot_img = image[y1:y2,x1:x2]
    spot_img = cv2.resize(spot_img, None, fx=2.0, fy=2.0)
    spot_id = spot_pos[spot]
    filename = 'spot' + str(spot_id) + '.jpg'
    print(spot_img.shape, filename, (x1, x2, y1, y2))
    cv2.imwrite('../../data/cnn_dataSpot/cnn_data'+ filename, spot_img)



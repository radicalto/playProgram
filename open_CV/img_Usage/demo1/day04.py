import numpy as np
import cv2
#*对图像添加高斯噪声
def add_gauss_noise(image, mean=0, val=0.01):
    size = image.shape
    # 对图像归一化处理
    image = image / 255
    gauss = np.random.normal(mean, val**0.05, size)
    print(gauss)
    image = image + gauss
    print(image)
    return image
# np.random.normal(loc=0.0, scale=1.0, size=None)
'''
loc(float):此概率分布的均值(对应着整个分布的中心centre
scale(float):此概率分布的标准差（对应于分布的宽度.scale越大.图形越矮胖;scale越小.图形越瘦高)
size(int or tuple of ints):输出的shape.默认为None.只输出一个值
'''
# 函数作用：
# 返回一组符合高斯分布的概率密度随机数
src = cv2.imread("../../image/cao.jpg")
print(src)
gauss_Src = add_gauss_noise(src)
cv2.imshow("gauss",gauss_Src)
cv2.drawContours()
cv2.waitKey(0)
cv2.destroyAllWindows()

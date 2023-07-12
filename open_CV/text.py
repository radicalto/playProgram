import numpy as np
from PIL import  Image
import cv2
src = cv2.imread('./image/1.jpg',0)
PIL_img = Image.open('./image/1.jpg').convert('L')
img_numpy = np.array(PIL_img,'uint8')
print(img_numpy)
print(src)

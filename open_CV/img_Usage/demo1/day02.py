import numpy as np
import cv2
img = np.array(
       [[[230, 225, 224],
        [230, 225, 224],
        [230, 225, 224],
        [229, 224, 223],
        [227, 222, 221]],

       [[231, 226, 225],
        [230, 225, 224],
        [230, 225, 224],
        [230, 225, 224],
        [229, 224, 223]],

       [[231, 226, 225],
        [231, 226, 225],
        [231, 226, 225],
        [231, 226, 225],
        [230, 225, 224]],

       [[230, 225, 224],
        [231, 226, 225],
        [231, 226, 225],
        [231, 226, 225],
        [230, 225, 224]],

       [[231, 226, 225],
        [232, 227, 226],
        [231, 226, 225],
        [231, 226, 225],
        [230, 225, 224]]], dtype=np.uint8)

mask = np.zeros([5, 5], dtype=np.uint8)+100
print("mask: ",mask)
mask[1]=[1,0,-1,1,0]
print("mask[1]: ",mask)
mask[2] = [-2, -1, 0, 1, 2]
print("mask[2]: ",mask)
img2 = cv2.add(img, img, mask=mask)
print("mask_img: ",img2)

"""mask[1]:  
[[  0   0   0   0   0]
 [  1   0 255   1   0]
 [  0   0   0   0   0]
 [  0   0   0   0   0]
 [  0   0   0   0   0]]

mask[2]:  
[[  0   0   0   0   0]
 [  1   0 255   1   0]
 [254 255   0   1   2]
 [  0   0   0   0   0]
 [  0   0   0   0   0]]

mask_img:  
[[[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]             

 [[255 255 255]
  [  0   0   0]
  [255 255 255]      [  1   0 255   1   0]
  [255 255 255]
  [  0   0   0]]

 [[255 255 255]
  [255 255 255]
  [  0   0   0]  [254 255   0   1   2]
  [255 255 255]
  [255 255 255]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]]"""
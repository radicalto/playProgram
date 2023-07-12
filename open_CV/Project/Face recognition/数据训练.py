import numpy as np

from Function import getImageAndLabels
import cv2
path = '../../tmp/'
#获取图片数组和id标签数组和姓名
ids,faces = getImageAndLabels(path)
#加载识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()
#训练
recognizer.train(faces,np.array(ids))
#保存文件
recognizer.write('../../trainer/trainer.yml')

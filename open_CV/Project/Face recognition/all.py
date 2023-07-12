import cv2
import os
import time
# 加在训练数据集文件
recognizer = cv2.face.LBPHFaceRecognizer_create()
# 加载数据
recognizer.read('../../trainer/trainer.yml')
# 名称
names=['宋潇阳','项超']
# 警报全局变量
warningtime=0
# 门
door=False
# md5加密
def md5(str):
    pass

# 警告模块
def warning():
    local_time = time.ctime()
    print('warning -- unknow people in '+local_time)

# 准备识别的图片
def face_detect_demo_finnal(door,img):
    if door==False:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(
            "D:\\pythonProject\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml")
        face = face_detector.detectMultiScale(gray_img,1.1)
        for x,y,w,h in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            ids,confidence = recognizer.predict(gray_img[y:y+h,x:x+w])
            # print(ids)
            if confidence > 80:
                global warningtime
                warningtime+=1
                if warningtime>10:
                    warning()
                    warningtime=0
                cv2.putText(img,'unknow',(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)
                print('错误,请走开')
            else:
                cv2.putText(img,str(names[ids-1]),(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)
                door=True
                print('开门')
            cv2.imshow("result",img)

cap = cv2.VideoCapture(0)
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_demo_finnal(door,frame)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break
    elif cv2.waitKey(1)& 0xFF == ord("s"):
        door = False
        print('门已关')

cap.release()
cv2.destroyAllWindows()

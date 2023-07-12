import cv2
cap = cv2.VideoCapture(0)
flag=1
num=1
while(cap.isOpened()):
    flag,frame = cap.read()
    cv2.imshow("Capture",frame)
    k = cv2.waitKey(0)
    if k==ord('s'):
        cv2.imwrite("D:\\pythonProject\\open_CV\\tmp\\"+str(num)+".jpg",frame)
        print("success to save"+str(num)+".jpg")
        num+=1
    else:
        break
cap.release()
cv2.destroyAllWindows()
from keras.saving.save import load_model
from Project.Func import cv_show
from parking_distinguish import spot_pos,image
import cv2
import numpy as np
weights_path = '../../data/car1.h5'
model = load_model(weights_path)

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'
predicted_images = np.copy(image)
overlay = np.copy(image)
# cv_show('predicted_images', predicted_images)
cnt_empty = 0
all_spots = 0
for spot in spot_pos.keys():
    all_spots += 1
    (x1, y1, x2, y2) = spot
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    spot_img = image[y1:y2, x1:x2]
    spot_img = cv2.resize(spot_img, (48, 48))

    # 预处理
    img = spot_img / 255.

    # 转换成4D tensor  keras要求
    spot_img = np.expand_dims(img, axis=0)

    # 用训练好的模型进行训练
    class_predicted = model.predict(spot_img)
    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]

    if label == 'empty':
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], -1)
        cnt_empty += 1
alpha = 0.5
cv2.addWeighted(overlay, alpha, predicted_images, 1 - alpha, 0, predicted_images)
cv2.putText(predicted_images, "Available: %d spots" % cnt_empty, (30, 95),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7, (255, 255, 255), 2)
cv2.putText(predicted_images, "Total: %d spots" % all_spots, (30, 125),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.7, (255, 255, 255), 2)
cv_show('predicted_images', predicted_images)


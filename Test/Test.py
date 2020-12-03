import numpy as np
from glob import glob
import cv2
import os
import numpy as np


def run_test(model, main_path, target_size):

    img_path_list = glob(main_path + "/*.jpg")
    class_label = {"c0":"safe driving", "c1":"texting-right", "c2":"talking on the phone - right", "c3" : "texting-left",
                   "c4":"talking on the phone-left", "c5":"operating the radio", "c6":"drinking","c7":"reaching behind",
                   "c8":"hair and makeup", "c9":"talking to passenger"}
    class_label = {idx:value for idx,value in enumerate(class_label.values())}
    print(class_label)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_input = cv2.resize(img, (target_size[1],target_size[0]))
        img_name = os.path.basename(img_path)
        img_input = cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB)
        img_input = img_input / 255
        img_input = np.array([img_input])
        test_result = model.predict(img_input)
        predicted_label = np.argmax(test_result)
        print(class_label[predicted_label])
        cv2.putText(img,class_label[predicted_label],(10,40), 2, 1,(0,0,255))
        cv2.imshow("img", img)
        cv2.waitKey(0)
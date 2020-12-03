import matplotlib.pyplot as plt
import cv2
import numpy as np


def draw_data_sampels(data, title):
    rows, cols = 3, 3
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)
    imgs, labels = data[0]
    class_idx = {value: key for key, value in data.class_indices.items()}
    class_label = {"c0":"safe driving", "c1":"texting-right", "c2":"talking on the phone - right", "c3" : "texting-left",
                   "c4":"talking on the phone-left", "c5":"operating the radio", "c6":"drinking","c7":"reaching behind",
                   "c8":"hair and makeup", "c9":"talking to passenger"}

    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        class_name = class_idx[np.argmax(labels[i])]
        img = imgs[i]
        plt.title(class_label[class_name])
        plt.imshow(img)
    plt.show()


def draw_test_result():
    None

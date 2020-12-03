from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from glob import glob
import os


def get_data_generator(main_path, target_size):
    path_train = f"{main_path}"
    # path_valid = f"{main_path}\\valid"
    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.05,
                                       zoom_range=[0.95, 1.05],
                                       width_shift_range=[-0.05, 0.05],
                                       height_shift_range=[-0.05, 0.05],
                                       rotation_range=5,
                                       validation_split=0.1)

    data_train = datagen_train.flow_from_directory(path_train,
                                                   subset="training",
                                                   target_size=target_size,
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=True)

    data_valid = datagen_train.flow_from_directory(path_train,
                                                   subset="validation",
                                                   target_size=target_size,
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=True)
    return data_train, data_valid


def get_test_data(main_path, target_size):

    img_path_list = glob(main_path + "/*.jpg")

    test_data = list()
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        img_name = os.path.basename(img_path)
        test_data.append((img, img_name))
    return test_data



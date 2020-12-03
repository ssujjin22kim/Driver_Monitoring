from Data import Data
from Model import Model
from Train import Train
from Test import Test
from Utils import Drawer


def train(data_train, data_valid, model_path):
    # check the train and validation data
    Drawer.draw_data_sampels(data_train, "data_train")

    # create a classification Model
    model = Model.create_model(classes=len(data_train.class_indices), input_size=data_train.image_shape)

    # run training
    history = Train.run_train(model=model, model_path = model_path, data_train=data_train, epochs=10, data_valid=data_valid)


def test(data_path, model_path, target_size):

    model = Model.load_model(model_path)
    Test.run_test(model, data_path, target_size)


if __name__ == '__main__':

    data_path_train = r"D:\Dataset\state-farm-distracted-driver-detection\imgs\train"
    data_path_test = r"D:\Dataset\state-farm-distracted-driver-detection\imgs\test"
    model_path = r"./trained_model.hdf5"
    target_size = (240, 320)

    mode = "train"

    if mode is "train":
        # get train and validation data
        data_train, data_valid = Data.get_data_generator(data_path_train, target_size)
        train(data_train, data_valid, model_path)
    elif mode is "test":
        test(data_path_test, model_path, target_size)

import tensorflow.keras as keras
from tensorflow.keras.models import load_model


def create_conv_block(filter_num, inputs):
    outputs = keras.layers.BatchNormalization()(inputs)
    outputs = keras.layers.Conv2D(filter_num, 3, activation="relu")(outputs)
    outputs = keras.layers.MaxPooling2D()(outputs)
    return outputs


def create_model(classes=12, input_size=(256,256,3)):
    # back bone model
    bb_model = keras.applications.EfficientNetB0(include_top=False)
    bb_model.trainable=False

    inputs = keras.layers.Input(shape=input_size)
    # outputs = bb_model(inputs)
    outputs = create_conv_block(64, inputs)
    outputs = create_conv_block(128, outputs)
    outputs = create_conv_block(256, outputs)
    outputs = create_conv_block(512, outputs)
    outputs = create_conv_block(512, outputs)

    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(1024, activation="relu")(outputs)
    outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def load_trained_model(model_path):
    return load_model(model_path)


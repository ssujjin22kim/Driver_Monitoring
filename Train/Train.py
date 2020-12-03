from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd


def save_train_history(history):
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(f)


def run_train(model, model_path, data_train, data_valid, epochs=10):
    callbacks = [ModelCheckpoint(model_path, verbose=1, save_best_only=True)]
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(data_train,  validation_data=data_valid, epochs=epochs, callbacks=callbacks, validation_steps=10)
    save_train_history(history)
    return history

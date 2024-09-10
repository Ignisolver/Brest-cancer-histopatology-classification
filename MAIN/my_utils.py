import numpy as np

from pathlib import Path

from keras.src import regularizers
from keras.src.optimizers.sgd import SGD
from matplotlib import pyplot as plt
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.src.optimizers.adam import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adamax


def freeze_all(model, trainable=False):
    for layer in model.layers:
        layer.trainable = trainable


def set_trainable(model, layers, trainable):
    for layer_nr in layers:
        model.layers[layer_nr].trainable = trainable


def add_NN(model, n_layers=3, dropout=0,
           normalization=False, units=512,
           out_units=1, activation='sigmoid',
           ):
    new_model = Sequential()
    new_model.add(model)
    # new_model.add(Flatten())
    # new_model.add(GlobalAveragePooling2D())
    new_model.add(GlobalMaxPool2D())
    if normalization:
        new_model.add(BatchNormalization())
    if dropout:
        new_model.add(Dropout(dropout))
    for _ in range(n_layers):
        new_model.add(Dense(units=units, activation=activation))
        if normalization:
            new_model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),)
        if dropout:
            new_model.add(Dropout(dropout))
    new_model.add(Dense(units=out_units, activation=activation))
    return new_model

def compile_model(model, lr=0.0001, loss='binary_crossentropy', metrics=None):
    if metrics is None:
        metrics = ['accuracy']
    model.compile(
        # optimizer=SGD(learning_rate=lr),
        optimizer=Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )


def get_checkpoint_callback(path):
    model_checkpoint = ModelCheckpoint(
        path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    return model_checkpoint


def get_early_stopping_callback(patience, start):
    es = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=start,
    )
    return es

def get_scheduler_callback(fcn):
    sch = LearningRateScheduler(fcn)
    return sch

def get_image_generator(path, preprocess_img_fcn,
                        val_split: float=None, test=False, batch_size=32,
                        class_mode='binary',
                        size=(50, 50)
                        ):

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
    if test:
        idg = ImageDataGenerator(preprocessing_function=preprocess_img_fcn)
        test_generator = idg.flow_from_directory(
            path,
            target_size=size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False)
        return test_generator
    else:
        idg = ImageDataGenerator(preprocessing_function=preprocess_img_fcn,
                                 validation_split=val_split)

        train_generator = idg.flow_from_directory(
            path,  # Folder z obrazami treningowymi
            target_size=size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True,
            subset='training')

        validation_generator = idg.flow_from_directory(
            path,  # Folder z obrazami walidacyjnymi
            target_size=size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset='validation')

        return train_generator, validation_generator


def save_model(model, path: Path, name):
    path = path/name
    model.save(path)


def plot_history(history, epoch_nr):
    H=history
    N = epoch_nr
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylim([0.4, 1.05])
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()
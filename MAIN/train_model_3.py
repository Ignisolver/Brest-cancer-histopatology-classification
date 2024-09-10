from keras.src.saving.saving_api import load_model

from my_constans import *
from loaded_models import models
from my_utils import *


def train_model_3(model_base_name, suffix, data_path, model=None, images=None):
    if model is None:
        model = models[model_base_name]["model"]
    preproces_fcn = models[model_base_name]["func"]
    name = model_base_name + suffix
    img_size = (50, 50)
    model = model(
        input_shape=(*img_size, 3),
        weights="imagenet",
        include_top=False,
    )
    # freeze_all(model)
    model = add_NN(model,
                   n_layers=0,
                   dropout=0.4,
                   normalization=True,
                   units=256,
                   out_units=1,
                   activation="sigmoid")
    compile_model(model, lr=0.0001, loss='binary_crossentropy')

    def scheduler_fcn(epoch, lr):
        e_1, e_2 = 2, 10
        if epoch < e_1:
            lr = 0.0001
        elif e_1 <= epoch < e_2:
            lr = 0.00001
        else:
            lr = 0.000001
        # with open("learning_rate.txt") as file:
        #     lr = float(file.readline())
        return lr
    scheduler = get_scheduler_callback(scheduler_fcn)
    es = get_early_stopping_callback(
        patience=6,
        start=15)
    checkpoint = get_checkpoint_callback(f'./models/{name}')
    callbacks = [
                 checkpoint,
                 # es,
                 scheduler
    ]
    t_gen, v_gen = get_image_generator(data_path,
                                       preproces_fcn,
                                       val_split=0.0275,
                                       test=False,
                                       batch_size=64,
                                       # class_mode='categorical',
                                       class_mode='binary',
                                       size=img_size)
    model.summary()

    history = model.fit(t_gen,
                  epochs=20,
                  validation_data=v_gen,
                  callbacks=callbacks)
    return history


if __name__ == "__main__":
    data_path = DATA_PATH / "train"
    model_base_name = "MobileNetV2"
    suffix = "_max_pull.keras"
    train_model_3(model_base_name, suffix, data_path)

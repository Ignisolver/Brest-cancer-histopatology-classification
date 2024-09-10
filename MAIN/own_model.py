from keras import Sequential
from keras.src.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.src.optimizers import Adam
from loaded_models import preprocess_input_resnet_v2
from MAIN.my_constans import DATA_PATH
from MAIN.my_utils import get_image_generator, get_scheduler_callback, get_checkpoint_callback

model = (Sequential
         ())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='softmax'))

model.compile(Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
data_path = DATA_PATH.parent / "ready" / "train"
t_gen, v_gen = get_image_generator(data_path,
                                       preprocess_input_resnet_v2,
                                       val_split=0.01,
                                       test=False,
                                       batch_size=40,
                                       class_mode='binary',
                                       size=(50, 50))


def scheduler_fcn(epoch, lr):
    # e_1, e_2 = 6, 10
    # if epoch < e_1:
    #     lr = 0.0001
    # elif e_1 <= epoch < e_2:
    #     lr = 0.00001
    # else:
    #     lr = 0.000001
    with open("learning_rate.txt") as file:
        lr = float(file.readline())
    return lr


scheduler = get_scheduler_callback(scheduler_fcn)
name = "Own.keras"
checkpoint = get_checkpoint_callback(f'./models/{name}')
model.fit(t_gen,
                  epochs=500,
                  validation_data=v_gen,
                  steps_per_epoch=1_00,
                  callbacks=[scheduler])

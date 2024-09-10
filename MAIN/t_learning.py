#%%
from keras import losses
from keras.src.saving.saving_api import load_model
import sys

from MAIN.loaded_models import models
from MAIN.my_constans import DATA_PATH
from MAIN.my_utils import get_image_generator, add_NN, compile_model, get_scheduler_callback, \
    get_early_stopping_callback, get_checkpoint_callback
from MAIN.train_model import train_model

sys.path.extend(['C:\\STUDIA\\Magisterka\\MAIN'])
#%%
model_name = "VGG16_best_92_from_0.keras"
model_teacher = load_model(f'C:\STUDIA\Magisterka\MAIN\models\{model_name}')
#%%
func = models[model_name.split('_')[0]]["func"]
t_gen, v_gen = get_image_generator(DATA_PATH.parent / "medium_10_000" / "train", func,
                             val_split=0.01,
                             test=False,
                             batch_size=32,
                             class_mode='binary',
                             size=(50,50)
                             )
#%%
predictions_t = model_teacher.predict(t_gen)
predictions_v = model_teacher.predict(v_gen)
#%%
predictions_t = list(map(lambda x: x[0], predictions_t))
predictions_v = list(map(lambda x: x[0], predictions_v))
#%%
t_gen.classes = predictions_t
v_gen.classes = predictions_v
#%%
from keras import losses
model_base_name = "VGG16"
model = models[model_base_name]["model"]
preproces_fcn = models[model_base_name]["func"]
name = model_base_name + "_t_learning"
img_size = (50, 50)
model = model(
    input_shape=(*img_size, 3),
    weights="imagenet",
    include_top=False,
)
# freeze_all(model)
model = add_NN(model,
               n_layers=0,
               dropout=0.2,
               normalization=True,
               units=256,
               out_units=1,
               activation="sigmoid")
compile_model(model, lr=0.0001,
              loss='mean_squared_error')
def scheduler_fcn(epoch, lr):
    e_1, e_2 = 6, 10
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
    patience=5,
    start=7)
checkpoint = get_checkpoint_callback(f'./models/{name}')
callbacks = [
             # checkpoint,
             # es,
             scheduler
]
model.summary()

history = model.fit(t_gen,
              epochs=50,
              validation_data=v_gen,
              steps_per_epoch=1_00,
              callbacks=callbacks)
#%%
predictions = model.predict(v_gen)
#%%
print(predictions)
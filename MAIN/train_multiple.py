#%%
import os
import shutil
import sys

from MAIN.train_model import train_model

sys.path.extend(['C:\\STUDIA\\Magisterka\\MAIN'])
from create_dataset import split_and_move_images
from my_constans import DATA_PATH
# from train_model import train_model
#%%
train_dataset_size = 15_000
n_models = 15
model_base_name = "MobileNetV2"
training_dir_path = DATA_PATH.parent / "multiple_train"
# os.mkdir(training_dir_path)
for i in range(n_models):
    folder_path = training_dir_path / str(i)
    split_and_move_images(folder_path, 0, train_dataset_size, fcn=shutil.copy)

#%%
for i in range(n_models):
    folder_path = training_dir_path / str(i)
    train_model(model_base_name, f"{i}_multiple_{train_dataset_size}.keras", folder_path / "train")

#%%
for i in range(n_models):
    folder_path = training_dir_path / str(i)
    split_and_move_images(DATA_PATH, None, None, folder_path)
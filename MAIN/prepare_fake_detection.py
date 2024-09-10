import os
import shutil
from pathlib import Path

from MAIN.create_dataset import split_and_move_images
from MAIN.my_constans import DATA_PATH, MODEL_DST
from MAIN.test_model import test_model
from MAIN.train_model import train_model
from augmentation import Augmentator
from extender import Extender
from info import DatasetInformer

# 1 - ok, 0 - miss
data_path = DATA_PATH / "fake_detection" / "train_base" / "train"
model_base_name = "MobileNetV2"
suffix = "_base.keras"
model_dst = MODEL_DST / "fakes"
# train_model(model_base_name, suffix, data_path)
# test on which model miss
model_path = model_dst / (model_base_name + suffix)

model_path = Path(r"C:\STUDIA\Magisterka\MAIN\models\MobileNetV2_max_pull_93.keras")
test_path = DATA_PATH / "fake_detection" / "set_fakes" / "test"
result, images = test_model(model_path, model_base_name, test_path, ret_images=True)
result = list(zip(result["summary"], images.names))
fakes = {"0": [], "1": []}
for (true, pred, _), name in result:
    if true != pred:
        fakes["0"].append((true, name))
    else:
        fakes["1"].append((true, name))

def move_to_dir(names, dir_path):
    for true, name in names:
        shutil.copy(data_path / true / name, dir_path/name)

move_to_dir(fakes["0"], test_path / "0")
move_to_dir(fakes["1"], test_path / "1")
input("Press")
# Augment miss and reduce not miss
di = DatasetInformer(test_path)
print(di)
ratio = di.percentage_abundance["0"]/di.percentage_abundance["1"]
print("RATIO 0 / 1:", ratio)
input("Press")
augmentator1 = Augmentator([1])
augmentator1.add_function(augmentator1.mirror_image(type_=0), stack_key=1)
augmentator2 = Augmentator([2])
augmentator2.add_function(augmentator2.mirror_image(type_=1), stack_key=2)
augmentator3 = Augmentator([3])
augmentator3.add_function(augmentator3.mirror_image(type_=2), stack_key=3)
os.mkdir(test_path / "ext")
extender1 = Extender(test_path / "0", test_path / "ext", augmentator3)
extender1.extend_class("0", 2)
extender2 = Extender(test_path / "0", test_path / "ext", augmentator2)
extender2.extend_class("0", 2)
extender3 = Extender(test_path / "0", test_path / "ext", augmentator1)
extender3.extend_class("0", float(input("pass ratio")))
for name in (test_path / "ext").iterdir():
    shutil.move(name, test_path / "0" / name.name)
os.rmdir(test_path / "ext")
os.rename(DATA_PATH / "fake_detection" / "set_fakes" / "test", DATA_PATH / "fake_detection" / "set_fakes" / "train")

target_folder_path = DATA_PATH / "fake_detection" / "set_fakes" / "train_test"
split_and_move_images(target_folder_path, 0, 10_000, fcn=shutil.move, source_folder=DATA_PATH / "fake_detection" / "set_fakes")
shutil.move(DATA_PATH / "fake_detection" / "set_fakes" / "train_test" / "train",
            DATA_PATH / "fake_detection" / "set_fakes" / "test")
input("Press")
# Train some model to detect miss and ok
test_path = DATA_PATH / "fake_detection" / "set_fakes" / "test"
model_base_name = "MobileNetV2"
suffix = "_detect_fakes.keras"
model_dst = MODEL_DST / "fakes"
train_model(model_base_name, suffix, test_path)
model_path = model_dst / (model_base_name+suffix)
result = test_model(model_path, test_path)
print(result["accuracy"])
input("Press")
# Split test dataset into miss and ok
# 1 - ok, 0 - miss
model_path = model_dst / (model_base_name + suffix)
test_path = DATA_PATH / "ready" / "test"
result, images = test_model(model_path, model_base_name, test_path, ret_images=True)
result = list(zip(result["summary"], images.names))
fakes = {"0": [], "1": []}
for (true, pred, _), name in result:
    if pred == 0:
        fakes["0"].append(name)
    elif pred == 1:
        fakes["1"].append(name)
    else:
        raise RuntimeError("change to string")
final_test_path = Path(r"C:\STUDIA\data\fake_detection\miss_not_miss")

for name in fakes["0"]:
    try:
        shutil.copy(DATA_PATH / "test" / "0" / name, final_test_path / "fake" / "test" / "0")
    except FileNotFoundError:
        shutil.copy(DATA_PATH / "test" / "1" / name, final_test_path / "fake" / "test" / "1")

for true, name in fakes["1"]:
    try:
        shutil.copy(DATA_PATH / "test" / "0" / name, final_test_path / "ok" / "test" / "0")
    except FileNotFoundError:
        shutil.copy(DATA_PATH / "test" / "1" / name, final_test_path / "ok" / "test" / "1")

# Test pretrained models on miss and ok







# # Get 80 000 images for train
# import os
# import shutil
# from pathlib import Path

# from tqdm import tqdm
#
# from MAIN.create_dataset import split_and_move_images
#
# train_folder = Path(r"C:\STUDIA\data\fake_detection\train_base")
# # split_and_move_images(train_folder, 0, 40_000, fcn=shutil.copy)
# # exit()
# # Train some models on train images
# ...
# # Get 200 000 images to train mis not miss
# fakes_detection_folder = Path(r"C:\STUDIA\data\fake_detection\set_fakes")
# # split_and_move_images(fakes_detection_folder, 0, 115_390, fcn=shutil.copy)
# train_names = list((train_folder / "train" / "0").iterdir())
# train_names.extend(list((train_folder / "train" / "1").iterdir()))
# base_path = fakes_detection_folder / "train" / "0"
# for train_image_path in tqdm(train_names):
#     path = base_path / train_image_path.name
#     try:
#         os.remove(path)
#     except FileNotFoundError:
#         continue
# base_path = fakes_detection_folder / "train" / "1"
# for train_image_path in tqdm(train_names):
#     path = base_path / train_image_path.name
#     try:
#         os.remove(path)
#     except FileNotFoundError:
#         continue
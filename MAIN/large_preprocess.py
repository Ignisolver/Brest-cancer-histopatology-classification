import numpy as np
from tqdm import tqdm
from MAIN.my_constans import DATA_PATH
import os
import shutil
from pathlib import Path

from MAIN.create_dataset import split_and_move_images
from augmentation import Augmentator
from dataset_preprocessing import ArchivePreprocessor
from extender import Extender
from images_preprocessing import ImagePreprocessor
from training_preper import prepare_training

V = 2
N = 5000
from info import DatasetInformer

ArchivePreprocessor(Path("C:/STUDIA/data/backup/archive"), Path("C:/STUDIA/data/flatten"), "D0").preprocess_dataset()
path = DATA_PATH / "preprocessed"

# names = path.iterdir()
#
# names_0 = filter(lambda x: x.name[0] == "0", names)
# names_1 = filter(lambda x:  x.name[-6] == '-', names)
# names_1 = list(names_1)
# print(names_1, len(names_1))
# n = 277_524 - 2 * 78_786

# names_0 = np.random.choice(list(names_0), n, replace=False)
# for name in tqdm(names_1):
#     os.remove(name)
# for name in tqdm(names_0):
#     os.remove(name)


# di = DatasetInformer(Path(f"C:/STUDIA/data/preprocessed"))
# print(di)
# ratio = di.percentage_abundance["0"]/di.percentage_abundance["1"]
# print("RATIO 0 / 1:", ratio)
# #
# ip = ImagePreprocessor(Path(f"C:/STUDIA/data/preprocessed"), Path(f"C:/STUDIA/data/preprocessed"))
# ip.add_function(ip.equalize_rgb_histogram)
# ip.preprocess_images(True)


# augmentator1 = Augmentator([1])
# augmentator2 = Augmentator([2])
# augmentator3 = Augmentator([3])
# augmentator3.add_function(augmentator3.mirror_image(type_=2), stack_key=3)
# augmentator2.add_function(augmentator2.mirror_image(type_=1), stack_key=2)
# augmentator1.add_function(augmentator1.mirror_image(type_=0), stack_key=1)
# extender3 = Extender(Path(f"C:/STUDIA/data/preprocessed"), Path(f"C:/STUDIA/data/ext"), augmentator3)
# extender3.extend_class("1", 2)
# extender3 = Extender(Path(f"C:/STUDIA/data/preprocessed"), Path(f"C:/STUDIA/data/ext"), augmentator1)
# extender3.extend_class("1", min(ratio-1, 2))
# extender3 = Extender(Path(f"C:/STUDIA/data/preprocessed"), Path(f"C:/STUDIA/data/ext"), augmentator2)
# extender3.extend_class("1", min(ratio-2, 2))

# n = 119952
# for file in Path("C:/STUDIA/data/ext").iterdir():
#     shutil.move(file, Path("C:/STUDIA/data/preprocessed") / file.name)
#     n-=1
#     if n == 0:
#         break
# os.rmdir(Path("C:/STUDIA/data/ext"))

# prepare_training(Path(f"C:/STUDIA/data/preprocessed"), Path(f"C:/STUDIA/data/ready_v_small/train"))

target_folder_path = Path(f"C:/STUDIA/data/ready_v_small_test")
split_and_move_images(target_folder_path, 0, N, fcn=shutil.move, source_folder=Path(f"C:/STUDIA/data/ready_v_small"))

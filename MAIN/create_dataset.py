import os
import random
import shutil
from functools import cache

from tqdm import tqdm

from MAIN.my_constans import DATA_PATH

@cache
def get_lists(source_folder):
    test_files_0 = os.listdir(os.path.join(source_folder, "test", "0"))
    test_files_1 = os.listdir(os.path.join(source_folder, "test", "1"))
    train_files_0 = os.listdir(os.path.join(source_folder, "train", "0"))
    train_files_1 = os.listdir(os.path.join(source_folder, "train", "1"))
    return test_files_0, test_files_1, train_files_0, train_files_1

def split_and_move_images(
        target_folder,
        num_test_images=None,
        num_train_images=None,
        source_folder=None,
        fcn=os.rename
        ):
    # Utworzenie nowej struktury katalogów
    if source_folder is None:
        source_folder = DATA_PATH
    if not target_folder.exists():
        os.makedirs(os.path.join(target_folder, "test", "0"))
        os.makedirs(os.path.join(target_folder, "test", "1"))
        os.makedirs(os.path.join(target_folder, "train", "0"))
        os.makedirs(os.path.join(target_folder, "train", "1"))
    # Listy plików w folderach 0 i 1
    test_files_0, test_files_1, train_files_0, train_files_1 = get_lists(source_folder)
    if num_test_images is None:
        num_test_images = len(test_files_0)
    if num_train_images is None:
        num_train_images = len(train_files_0)
    # Losowe wybranie plików do przeniesienia
    random_test_files_0 = random.sample(test_files_0, num_test_images)
    random_test_files_1 = random.sample(test_files_1, num_test_images)
    random_train_files_0 = random.sample(train_files_0, num_train_images)
    random_train_files_1 = random.sample(train_files_1, num_train_images)
    # Przenoszenie zdjęć do folderu testowego
    for file in tqdm(random_test_files_0):
      fcn(os.path.join(source_folder, "test", "0", file), os.path.join(target_folder, "test", "0", file))
    for file in tqdm(random_test_files_1):
      fcn(os.path.join(source_folder, "test", "1", file), os.path.join(target_folder, "test", "1", file))
    # Przenoszenie zdjęć do folderu treningowego
    for file in tqdm(random_train_files_0):
      fcn(os.path.join(source_folder, "train", "0", file), os.path.join(target_folder, "train", "0", file))
    for file in tqdm(random_train_files_1):
      fcn(os.path.join(source_folder, "train", "1", file), os.path.join(target_folder, "train", "1", file))

if __name__ == "__main__":
    # Użycie funkcji

    # split_and_move_images(DATA_PATH, None, None, source_folder=src_folder_path)
    folder_name = "large_2_test"
    target_folder_path = DATA_PATH.parent / folder_name
    split_and_move_images(target_folder_path, 0, 2_500, fcn=shutil.move, source_folder=DATA_PATH.parent / "l_2_ready")
    folder_name = "large_3_test"
    target_folder_path = DATA_PATH.parent / folder_name
    split_and_move_images(target_folder_path, 0, 2_000, fcn=shutil.move, source_folder=DATA_PATH.parent / "l_3_ready")

import shutil
from pathlib import Path

from tqdm import tqdm
from PIL import Image

from MAIN.create_dataset import split_and_move_images
from MAIN.my_constans import DATA_PATH
from image_label import ImageLabel
from images_preprocessing import ImagePreprocessor
from info import DatasetInformer
from utils import clear_folder


def prepare_training(src_path: Path, dest_path: Path):
    dest_path.mkdir(parents=True, exist_ok=True)

    di = DatasetInformer(src_path)
    classes = list(di.get_balance().keys())
    for class_name in classes:
        class_folder = dest_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(src_path.iterdir()), desc="Moving images", unit=" image"):
        class_id = ImageLabel(path=img_path).class_id
        shutil.copy(img_path, dest_path / class_id)


if __name__ == "__main__":
    # clear_folder(Path("../data/ready/train"))
    ip = ImagePreprocessor(Path("../data/large_2"), Path("../data/large_3"))
    ip.add_function(ip.equalize_rgb_histogram)
    ip.preprocess_images(True)

    ip = ImagePreprocessor(Path("../data/large_3"), Path("../data/large_3"))
    ip.add_function(ip.equalize_rgb_histogram)
    ip.preprocess_images(True)

    prepare_training(Path("../data/large_2"), Path("../data/large_2/train"))
    prepare_training(Path("../data/large_3"), Path("../data/large_3/train"))

    folder_name = "large_2_test"
    target_folder_path = DATA_PATH.parent / folder_name
    split_and_move_images(target_folder_path, 0, 3_000, fcn=shutil.move, source_folder=DATA_PATH.parent / "large_2")

    folder_name = "large_3_test"
    target_folder_path = DATA_PATH.parent / folder_name
    split_and_move_images(target_folder_path, 0, 2_500, fcn=shutil.move, source_folder=DATA_PATH.parent / "large_3")
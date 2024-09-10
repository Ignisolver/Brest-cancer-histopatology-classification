import shutil
import tqdm

from abc import ABC, abstractmethod
from pathlib import Path

from image_label import ImageLabel


class DatasetPreprocessorInterface(ABC):
    def __init__(self, src_folder_path: Path, dst_folder_path: Path):
        self.src_folder_path = src_folder_path
        self.dst_folder_path = dst_folder_path

    @abstractmethod
    def preprocess_dataset(self):
        pass


class ArchivePreprocessor(DatasetPreprocessorInterface):

    def __init__(self, src_folder_path, dst_folder_path, dataset_id: str):
        super().__init__(src_folder_path, dst_folder_path)
        self.dataset_id = dataset_id

    def preprocess_dataset(self):
        id_ = 0
        for patient_folder_path in tqdm.tqdm(list(self.src_folder_path.iterdir()), position=0, unit="patient"):
            for label_folder_path in tqdm.tqdm(list(patient_folder_path.iterdir()), position=1, unit="class", leave=False):
                for image_path in tqdm.tqdm(list(label_folder_path.iterdir()), position=2, unit='image', leave=False):
                    file_name = image_path.name
                    image_class = file_name[-5]
                    # 8863_idx5_x101_y1201_class0
                    x, y = file_name.split("_")[2:4]
                    x = int((int(x[1:])-1)/50)
                    y = int((int(y[1:])-1)/50)
                    new_name = ImageLabel.create_label_str(str(image_class), patient_folder_path.name + "-" + str(x) + "-" + str(y), str(id_)) + ".png"
                    dest_path = self.dst_folder_path / new_name
                    shutil.move(image_path, dest_path)
                    id_ += 1

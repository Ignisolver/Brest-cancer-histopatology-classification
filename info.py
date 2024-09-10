import json

import numpy as np

from pathlib import Path
from PIL import Image


class DatasetInformer:
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.paths = list(self.folder_path.iterdir())
        self.size = len(self.paths)
        self.abundance = None
        self.percentage_abundance = None
        self.shape = None
        self.mode = None

    def get_balance(self):
        classes = {}
        for path in self.paths:
            class_name = path.name.split("_")[0]
            try:
                classes[class_name] += 1
            except KeyError:
                classes[class_name] = 1
        self.abundance = classes
        return classes

    def get_balance_ratio(self):
        if not self.abundance:
            self.get_balance()
        ratios = {}
        for class_name, class_size in self.abundance.items():
            ratios[class_name] = class_size/self.size
        self.percentage_abundance = ratios
        return ratios

    def get_image_info(self):
        image = Image.open(self.paths[0])
        image_array = np.array(image)
        self.shape = image_array.shape
        self.mode = image.mode
        return self.shape

    def __repr__(self):
        str_ = ""
        if not self.percentage_abundance:
            self.get_balance_ratio()
        if not self.shape:
            self.get_image_info()
        str_ += "Dataset information:" + "\n"
        str_ += "PATH: " + str(self.folder_path) + "\n"
        str_ += "FOLDER NAME: " + str(self.folder_path.name) + "\n"
        str_ += "DATASET SIZE: " + str(self.size) + "\n"
        str_ += "ABUNDANCE:" + "\n"
        balance = json.dumps(self.abundance, indent=4)
        str_ += balance + "\n"
        str_ += "PERCENTAGE ABUNDANCE:" + "\n"
        balance_ratio = json.dumps(self.percentage_abundance, indent=4, )
        str_ += balance_ratio + "\n"
        str_ += "IMAGE_SIZE: " + str(self.shape) + "\n"
        str_ += "IMAGE_MODE: " + str(self.mode) + "\n"

        return str_

if __name__ == "__main__":
    di = DatasetInformer(Path("../data/preprocessed"))
    print(di)

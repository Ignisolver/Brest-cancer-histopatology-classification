from pathlib import Path
from typing import List


class ImageLabel:
    def __init__(self, path: Path = None, label: str = None):
        self.path = path
        self.label = label
        self.extension = None
        if path is not None:
            self.label = path.name
        self.class_id = None
        self.image_id = None
        self.dataset_id = None
        self._parse_label()

    def _parse_label(self):
        data, extension = self.label.split('.')
        self.extension = extension
        try:
            class_id, image_id = data.split("_")  # todo change class_id, dataset_id, image_id = data.split("_")
        except:
            class_id, _, image_id = data.split("_")
        self.class_id = class_id
        self.dataset_id = str(1)  # todo changeself.dataset_id = dataset_id
        self.image_id = image_id

    def get_label_str(self):
        return self.create_label_str(self.class_id, self.dataset_id, self.image_id)

    def get_file_name_str(self):
        return self.get_label_str() + "." + self.extension

    @staticmethod
    def parse_paths(paths):
        image_labels = []
        for path in paths:
            i_l = ImageLabel(path)
            image_labels.append(i_l)
        return image_labels

    @staticmethod
    def extract_class_labels(labels: List["ImageLabel"], class_id):
        return list(filter(lambda x: x.class_id == class_id, labels))

    @staticmethod
    def create_label_str(class_id, dataset_id, image_id):
        return "_".join([class_id, dataset_id, image_id])

    @staticmethod
    def get_paths_from_labels(labels: List["ImageLabel"]):
        class_paths = list(map(lambda x: x.path, labels))
        return class_paths

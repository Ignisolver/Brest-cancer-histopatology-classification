from typing import List

import numpy as np

from itertools import cycle
from pathlib import Path

from tqdm import tqdm

from ImageOperator import ImageOperator
from augmentation import Augmentator
from image_label import ImageLabel


class Extender:
    def __init__(self, src_path: Path, dest_folder_path: Path, augmentator: Augmentator):
        self.src_path = src_path
        self.dest_folder_path = dest_folder_path
        self.augmentator = augmentator

    def extend_class(self, class_id, factor):
        n, paths, paths_to_extend = self._get_paths_to_extend(class_id, factor)
        if n:
            for src_im_path in tqdm(paths):
                keys_generator = cycle(self.augmentator.stacks_keys)
                for i in range(n):
                    augmentator_key = next(keys_generator)
                    self._augment_image(augmentator_key, src_im_path)
        keys_generator = cycle(self.augmentator.stacks_keys)
        for src_im_path in tqdm(paths_to_extend):
            augmentator_key = next(keys_generator)
            self._augment_image(augmentator_key, src_im_path)

    def _augment_image(self, augmentator_key, src_im_path):
        label = ImageLabel(path=src_im_path)
        label.image_id += "-" + str(augmentator_key)
        new_name = label.get_file_name_str()
        with ImageOperator(src_im_path, self.dest_folder_path, read=True, delete=False, new_name=new_name) as im_op:
            im_op.image = self.augmentator.augment_image(im_op.image, [augmentator_key], ret_raw=True)

    def _get_class_labels(self, class_id):
        all_labels = ImageLabel.parse_paths(self.src_path.iterdir())
        class_labels = ImageLabel.extract_class_labels(all_labels, class_id=class_id)
        return class_labels

    def _get_paths_to_extend(self, class_id, factor):
        class_labels = self._get_class_labels(class_id)
        class_size = len(class_labels)
        n_random = int((factor - int(factor)) * class_size)
        n_repeat = int(factor-1)
        paths = ImageLabel.get_paths_from_labels(class_labels)
        paths_to_extend = np.random.choice(paths, n_random, replace=False)
        return n_repeat, paths, paths_to_extend


# ia = Augmentator([1, 2])
#
# ia.add_function(ia.rotate_image(angle=10), stack_key=1)
# ia.add_function(ia.mirror_image(type_=0), stack_key=1)
# ia.add_function(ia.adjust_brightness(factor=2), stack_key=1)
#
# ia.add_function(ia.remove_fragment(width_factor=0.2, height_factor=0.2, n=1), stack_key=2)
# ia.add_function(ia.adjust_contrast(factor=0.5), stack_key=2)
#
# extender = Extender(Path('../data/test'), Path('../data'), ia)
if __name__ == '__main__':

    augmentator = Augmentator([1, 2, 3, 4, 5])
    augmentator.add_function(augmentator.mirror_image(type_=0), stack_key=1)

    augmentator.add_function(augmentator.mirror_image(type_=1), stack_key=2)

    augmentator.add_function(augmentator.mirror_image(type_=2), stack_key=3)

    augmentator.add_function(augmentator.rotate_image(angle=5), stack_key=4)
    augmentator.add_function(augmentator.stretch_image(factor_width=1.1, factor_height=1.2), stack_key=4)

    augmentator.add_function(augmentator.rotate_image(angle=-5), stack_key=5)
    augmentator.add_function(augmentator.stretch_image(factor_width=1.1, factor_height=1.2), stack_key=5)

    extender = Extender(Path("../data/preprocessed"), Path("../data/preprocessed"), augmentator)
    extender.extend_class("1", 2.4239617106685176)
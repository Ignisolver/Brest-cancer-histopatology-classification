import typing
from pathlib import Path

import cv2
import numpy as np

from PIL import ImageOps, Image
from tqdm import tqdm

from ExecutionPlanner import ExecutionPlanner, partial_decorator
from ImageOperator import ImageOperator
from utils import iter_over_images_in_folder, clear_folder


class ImagePreprocessor(ExecutionPlanner):
    def __init__(self, src_folder, dst_folder):
        super(ImagePreprocessor, self).__init__()
        self.src_folder = src_folder
        self.dst_folder = dst_folder

    def preprocess_images(self, delete):
        for image_path in tqdm(list(iter_over_images_in_folder(self.src_folder)), desc="Preprocessing images", unit=" image"):
            with ImageOperator(image_path, self.dst_folder, read=True, delete=delete) as image_operator:
                image_operator.image = self.execute(image_operator.image, ret_raw=True)

    @staticmethod
    def to_hsv(image):
        return image.convert('HSV')

    @staticmethod
    @partial_decorator
    def set_size(image, size_x_y):
        return image.resize(size_x_y)

    @staticmethod
    def to_grayscale(image):
        return ImageOps.grayscale(image)

    @staticmethod
    def equalize_histogram(image):
        return ImageOps.equalize(image)

    @staticmethod
    def equalize_histogram_2(image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_l_channel = clahe.apply(l_channel)

        # Merge the modified L channel with the original A and B channels
        modified_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

        # Convert the LAB image back to RGB color space
        final_image = cv2.cvtColor(modified_lab_image, cv2.COLOR_LAB2BGR)
        final_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

        return final_image
    @staticmethod
    def equalize_rgb_histogram(image):
        numpy_array = np.array(image)
        img_yuv = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        pil_image = Image.fromarray(img_output)
        return pil_image

if __name__ == '__main__':
    clear_folder(Path("../data/preprocessed"))
    ip = ImagePreprocessor(Path("../data/flatten"), Path("../data/preprocessed"))
    ip.add_function(ip.equalize_rgb_histogram)
    ip.preprocess_images(delete=False)

import numpy as np

from PIL import Image, ImageEnhance

from ExecutionPlanner import partial_decorator, ExecutionPlanner
from utils import display_image


class Augmentator(ExecutionPlanner):
    def __init__(self, stacks_keys):
        super(Augmentator, self).__init__(stacks_keys)

    def augment_image(self, image, specified_stacks, ret_raw=True):
        if ret_raw:
            return self.execute(image, specified_stacks=specified_stacks, ret_raw=ret_raw)
        else:
            return self.execute(image, specified_stacks=specified_stacks, ret_raw=ret_raw).values()

    @staticmethod
    @partial_decorator
    def rotate_image(image, angle):
        """Rotates the image by a specified angle (angle). The angle parameter is expressed in degrees."""
        return image.rotate(angle)

    @staticmethod
    @partial_decorator
    def mirror_image(image, type_=0):
        """Mirror image in specified axis"""
        if type_ == 0:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif type_ == 1:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif type_ == 2:
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    @partial_decorator
    def adjust_brightness(image, factor):
        """Adjusts the brightness of the image using the specified factor (factor).
         For factor > 1, the brightness increases; for factor < 1, the brightness decreases."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    @partial_decorator
    def adjust_contrast(image, factor):
        """Adjusts the contrast of the image using the specified coefficient (factor).
         For factor > 1, contrast increases; for factor < 1, contrast decreases."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    @partial_decorator
    def add_gaussian_noise(image, mean=0, std=25):
        """Adds Gaussian noise with the given mean (mean) and standard deviation (std).
         The mean and std parameters control the characteristics of the noise."""
        data = np.array(image)

        if len(data.shape) == 2:
            height, width = data.shape
            noise = np.random.normal(mean, std, (height, width))
            noisy_data = np.clip(data + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
        elif len(data.shape) == 3 and data.shape[2] in [1, 3]:
            noise = np.random.normal(mean, std, data.shape)
            noisy_data = np.clip(data + noise, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Incorrect image depth")

        return Image.fromarray(noisy_data)

    @staticmethod
    @partial_decorator
    def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """Adds noise of type salt and pepper. The salt_prob and pepper_prob
         parameters specify the probability of salt or pepper type pixels."""
        img_array = np.array(image)
        salt_mask = np.random.rand(*img_array.shape[:2]) < salt_prob
        pepper_mask = np.random.rand(*img_array.shape[:2]) < pepper_prob

        img_array[salt_mask] = 255
        img_array[pepper_mask] = 0

        return Image.fromarray(img_array)

    @staticmethod
    @partial_decorator
    def stretch_image(image, factor_width, factor_height):
        width, height = image.size

        zoomed_width = int(width / factor_width)
        zoomed_height = int(height / factor_height)

        left = (width - zoomed_width) // 2
        top = (height - zoomed_height) // 2
        right = left + zoomed_width
        bottom = top + zoomed_height

        zoomed_image = image.crop((left, top, right, bottom))
        zoomed_image = zoomed_image.resize((width, height))

        return zoomed_image

    @staticmethod
    @partial_decorator
    def remove_fragment(image, width_factor=0.2, height_factor=0.2, n=2):
        size_x = int(image.size[0] * width_factor)
        size_y = int(image.size[1] * height_factor)
        max_x_start = image.size[0] - size_x
        max_y_start = image.size[1] - size_y
        new_image = Image.new("RGB", image.size)
        new_image.paste(image, (0, 0))
        for i in range(n):
            start_x = np.random.randint(0, max_x_start)
            start_y = np.random.randint(0, max_y_start)
            dark = Image.new("RGB", (size_x, size_y))
            new_image.paste(dark, (start_x, start_y))
        return new_image

if __name__ == "__main__":
    image = Image.open("../data/preprocessed/0_1.png")
    ia = Augmentator([1, 2, 3])

    ia.add_function(ia.rotate_image(angle=10), stack_key=1)
    ia.add_function(ia.mirror_image(type_=0), stack_key=1)
    ia.add_function(ia.adjust_brightness(factor=2), stack_key=1)

    ia.add_function(ia.remove_fragment(width_factor=0.2, height_factor=0.2, n=1), stack_key=2)
    ia.add_function(ia.adjust_contrast(factor=0.5), stack_key=2)

    ia.add_function(ia.stretch_image(factor_width=0.9, factor_height=0.9), stack_key=3)


    imgs = ia.augment_image(image, [3], ret_raw=False)
    for img in imgs:
        display_image(img)


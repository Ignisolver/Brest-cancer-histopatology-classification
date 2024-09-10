import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def iter_over_images_in_folder(folder_path: Path):
    for image_path in folder_path.iterdir():
        yield image_path


def display_image(image: Image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def clear_folder(folder_path: Path):
    for path in tqdm(list(folder_path.iterdir()), desc=f"clearing: {folder_path}", unit="file"):
        path.unlink()

if __name__ == "__main__":
    clear_folder(Path(r"C:\STUDIA\data\preprocessed"))
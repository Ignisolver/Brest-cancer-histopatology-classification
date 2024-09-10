import os
import random

import cv2
from tqdm import tqdm

from MAIN.my_constans import DATA_PATH
from MAIN.test_model import test_model
from MAIN.train_model import train_model


def divide_image(image_path, n):
    # Wczytaj obraz
    image = cv2.imread(image_path)

    # Pobierz wymiary obrazu
    height, width, _ = image.shape

    # Sprawdź, czy obraz jest kwadratowy
    if height != width:
        print("Obraz nie jest kwadratowy. Wymagany jest kwadratowy obraz.")
        return False

    # Oblicz szerokość i wysokość małych kwadratów
    square_size = width // n

    images = []
    # Podziel obraz na n części i zapisz każdą
    for i in range(n):
        for j in range(n):
            small_square = image[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size]
            images.append(small_square)
    return images


def save_images(folder, images, name):
    for i, image in enumerate(images, 1):
        cv2.imwrite(os.path.join(folder, f'{name}_{i}.png'), image)


def make_train(size, mode, n, inp_path, out_path):
    # Ścieżka do folderu z obrazami
    input_folder = inp_path / mode
    output_path = out_path / mode
    num_images = n
    for folder in input_folder.iterdir():
        class_name = folder.name
        new_folder = output_path / class_name
        names = list(folder.iterdir())
        bad = 0
        chosen_names = random.sample(names, int(num_images/2))
        for nr, image_name in tqdm(list(enumerate(chosen_names))):
            images = divide_image(str(image_name), size)
            if images:
                save_images(new_folder, images, f"image_{class_name}-{nr}")
        print(bad)


def train():
    data_path = DATA_PATH.parent / "splitted_4" / "train"
    model_base_name = "MobileNetV2"
    suffix = "_small_4.keras"
    train_model(model_base_name, suffix, data_path)


def merge_results(result):
    images_result = {}
    for (true, predicted, predictions), name in result:
        img_name = name.split("_")[1]
        try:
            images_result[img_name][1].append(predicted)
            images_result[img_name][2].append(predictions)
        except:
            images_result[img_name] = [true, [predicted], [predictions]]

    results = []
    for key in images_result:
        result = 1 if sum(images_result[key][1])/len(images_result[key][1]) >= 0.5 else 0
        # result = 1 if sum(images_result[key][2])/len(images_result[key][1]) >= 0.5 else 0
        results.append(int(result == images_result[key][0]))


    print("Better_acc:", sum(results)/len(results))




def test():
    name = "MobileNetV2_max_pull_93.keras"
    data_path = DATA_PATH.parent / "splitted_4" / "test"
    results, images = test_model(name, data_path, ret_images=True)
    print("Base accuracy:", results["accuracy"])
    result = list(zip(results["summary"], images.filenames))
    merge_results(result)



if __name__ == "__main__":
    make_train(2, "train", 30_000, DATA_PATH.parent / "large_2", DATA_PATH.parent / "d")
    make_train(2, "test", 1_000)
    # train()
    test()

from dataclasses import dataclass
from pathlib import Path
import os
from typing import List, Tuple

from tqdm import tqdm

from MAIN.my_constans import DATA_PATH

VERSION = 2

def get_info_from_name(name: str):
    class_, patient_x_y, base_id_key = name[:-4].split("_")
    patient_id, x, y = tuple(map(int, patient_x_y.split('-')))
    # base_id, key = base_id_key
    return int(class_), base_id_key, x, y, patient_id

max_x_size = 0
max_y_size = 0
patients = set()

for image in Path(DATA_PATH.parent / "flatten").iterdir():
    try:
        _, base_id, x, y, patient_id = get_info_from_name(image.name)
    except:
        continue
    if x > max_x_size:
        max_x_size = x
    if y > max_y_size:
        max_y_size = y
    patients.update([patient_id])


tabs = {}
names_from_tabs = {}
for patient_id in list(patients):
    tab = [[None]*max_x_size for i in range(max_y_size)]
    tabs[patient_id] = tab
    names_from_tabs[patient_id] = []
for image in Path(DATA_PATH.parent / "flatten").iterdir():
    try:
        class_, base_id, x, y, patient_id = get_info_from_name(image.name)
    except:
        continue
    tabs[patient_id][y-1][x-1] = class_
    names_from_tabs[patient_id].append(image.name)

positions = dict.fromkeys(tabs, None)
coord = None
y_x = []
@dataclass
class Square:
    value: int
    y_x: Tuple[int, int]
positive = 0
negative = 0
for patient, tab in tabs.items():
    for y, row in enumerate(tab, 0):
        for x, el in enumerate(row, 0):
            try:
                if VERSION == 2:
                    elements = [el, tab[y+1][x+1], tab[y+1][x], tab[y][x+1]]
                if VERSION == 3:
                    elements = [tab[y][x], tab[y][x+1], tab[y][x+2],
                                tab[y+1][x], tab[y+1][x+1], tab[y+1][x+2],
                                tab[y+2][x], tab[y+2][x+1], tab[y+2][x+2]]
                if VERSION == 4:
                    elements = [
                        tab[y + 0][x + 0],
                        tab[y + 0][x + 1],
                        tab[y + 0][x + 2],
                        tab[y + 0][x + 3],
                        tab[y + 1][x + 0],
                        tab[y + 1][x + 1],
                        tab[y + 1][x + 2],
                        tab[y + 1][x + 3],
                        tab[y + 2][x + 0],
                        tab[y + 2][x + 1],
                        tab[y + 2][x + 2],
                        tab[y + 2][x + 3],
                        tab[y + 3][x + 0],
                        tab[y + 3][x + 1],
                        tab[y + 3][x + 2],
                        tab[y + 3][x + 3],
                    ]
                if VERSION == 5:
                    elements = [
                        tab[y + 0][x + 0],
                        tab[y + 0][x + 1],
                        tab[y + 0][x + 2],
                        tab[y + 0][x + 3],
                        tab[y + 0][x + 4],
                        tab[y + 1][x + 0],
                        tab[y + 1][x + 1],
                        tab[y + 1][x + 2],
                        tab[y + 1][x + 3],
                        tab[y + 1][x + 4],
                        tab[y + 2][x + 0],
                        tab[y + 2][x + 1],
                        tab[y + 2][x + 2],
                        tab[y + 2][x + 3],
                        tab[y + 2][x + 4],
                        tab[y + 3][x + 0],
                        tab[y + 3][x + 1],
                        tab[y + 3][x + 2],
                        tab[y + 3][x + 3],
                        tab[y + 3][x + 4],
                        tab[y + 4][x + 0],
                        tab[y + 4][x + 1],
                        tab[y + 4][x + 2],
                        tab[y + 4][x + 3],
                        tab[y + 4][x + 4],
                    ]
            except IndexError:
                continue
            if not (None in elements):
                y_x = ((x+1) * 50 + 1, (y+1) * 50 + 1)
                if sum(elements) == 0:
                    value = 0
                    negative += 1
                elif sum(elements) == VERSION**2:
                    value = 1
                    positive += 1
                else:
                    continue
                square = Square(value, y_x)
                try:
                    positions[patient].append(square)
                except AttributeError:
                    positions[patient] = [square]
print(positive, negative)
len_ = sum([(len(x) if x is not None else 0) for x in positions.values()])
print(len_)
sum_large = 0
for patient in positions:
    try:
        sum_large += len(positions[patient])
    except TypeError:
        pass

all_names = []
name_match = "{id}_idx5_x{y}_y{x}_class{class}.png"
for patient, positions in positions.items():
    if positions is None:
        continue
    for position in positions:
        if VERSION == 2:
            coords = [position.y_x,
                      (position.y_x[0]+50, position.y_x[1]+50),
                      (position.y_x[0]+50, position.y_x[1]),
                      (position.y_x[0], position.y_x[1]+50),
                      ]
        if VERSION == 3:
            coords = [(position.y_x[0], position.y_x[1]),
                      (position.y_x[0], position.y_x[1]+50),
                      (position.y_x[0], position.y_x[1]+100),
                      (position.y_x[0]+50, position.y_x[1]),
                      (position.y_x[0]+50, position.y_x[1]+50),
                      (position.y_x[0]+50, position.y_x[1]+100),
                      (position.y_x[0]+100, position.y_x[1]),
                      (position.y_x[0]+100, position.y_x[1]+50),
                      (position.y_x[0]+100, position.y_x[1]+100)]
        if VERSION == 4:
            coords = [(position.y_x[0], position.y_x[1]),
                      (position.y_x[0], position.y_x[1] + 50),
                      (position.y_x[0], position.y_x[1] + 100),
                      (position.y_x[0], position.y_x[1] + 150),
                      (position.y_x[0] + 50, position.y_x[1]),
                      (position.y_x[0] + 50, position.y_x[1] + 50),
                      (position.y_x[0] + 50, position.y_x[1] + 100),
                      (position.y_x[0] + 50, position.y_x[1] + 150),
                      (position.y_x[0] + 100, position.y_x[1]),
                      (position.y_x[0] + 100, position.y_x[1] + 50),
                      (position.y_x[0] + 100, position.y_x[1] + 100),
                      (position.y_x[0] + 100, position.y_x[1] + 150),
                      (position.y_x[0] + 150, position.y_x[1]),
                      (position.y_x[0] + 150, position.y_x[1] + 50),
                      (position.y_x[0] + 150, position.y_x[1] + 100),
                      (position.y_x[0] + 150, position.y_x[1] + 150),
                      ]
        if VERSION == 5:
            coords = [
                (position.y_x[0], position.y_x[1]),
                (position.y_x[0], position.y_x[1] + 50),
                (position.y_x[0], position.y_x[1] + 100),
                (position.y_x[0], position.y_x[1] + 150),
                (position.y_x[0], position.y_x[1] + 200),
                (position.y_x[0] + 50, position.y_x[1]),
                (position.y_x[0] + 50, position.y_x[1] + 50),
                (position.y_x[0] + 50, position.y_x[1] + 100),
                (position.y_x[0] + 50, position.y_x[1] + 150),
                (position.y_x[0] + 50, position.y_x[1] + 200),
                (position.y_x[0] + 100, position.y_x[1]),
                (position.y_x[0] + 100, position.y_x[1] + 50),
                (position.y_x[0] + 100, position.y_x[1] + 100),
                (position.y_x[0] + 100, position.y_x[1] + 150),
                (position.y_x[0] + 100, position.y_x[1] + 200),
                (position.y_x[0] + 150, position.y_x[1]),
                (position.y_x[0] + 150, position.y_x[1] + 50),
                (position.y_x[0] + 150, position.y_x[1] + 100),
                (position.y_x[0] + 150, position.y_x[1] + 150),
                (position.y_x[0] + 150, position.y_x[1] + 200),
                (position.y_x[0] + 200, position.y_x[1]),
                (position.y_x[0] + 200, position.y_x[1] + 50),
                (position.y_x[0] + 200, position.y_x[1] + 100),
                (position.y_x[0] + 200, position.y_x[1] + 150),
                (position.y_x[0] + 200, position.y_x[1] + 200),
            ]

        names = [(patient, position.value)]
        for coord in coords:
            names.append(name_match.format(**{"id": patient,
                                              "y": coord[0],
                                              'x': coord[1],
                                              "class": position.value}))
        all_names.append(names)

from PIL import Image
import os


def merge_images(folder_path, image_names, output_folder, output_name):
    # Create a list to hold the image parts
    images = []

    # Open and append each image part to the list
    for img_name in image_names:
        img_path = folder_path + "/" + img_name
        img = Image.open(img_path)
        images.append(img)
    # Determine the dimensions of the final image
    width, height = images[0].size
    final_width = width * VERSION
    final_height = height * VERSION

    # Create a new blank image with the calculated dimensions
    merged_image = Image.new('RGB', (final_width, final_height))

    # Paste each image part onto the blank image at the appropriate position
    if VERSION == 2:
        merged_image.paste(images[0], (0, 0))
        merged_image.paste(images[2], (width, 0))
        merged_image.paste(images[3], (0, height))
        merged_image.paste(images[1], (width, height))

    if VERSION == 3:
        merged_image.paste(images[0], (0, 0))
        merged_image.paste(images[3], (width, 0))
        merged_image.paste(images[6], (2 * width, 0))

        merged_image.paste(images[1], (0, height))
        merged_image.paste(images[4], (width, height))
        merged_image.paste(images[7], (2 * width, height))

        merged_image.paste(images[2], (0, 2 * height))
        merged_image.paste(images[5], (width, 2 * height))
        merged_image.paste(images[8], (2 * width, 2 * height))

    if VERSION == 4:
        merged_image.paste(images[0], (0 * width, 0 * height))
        merged_image.paste(images[1], (0 * width, 1 * height))
        merged_image.paste(images[2], (0 * width, 2 * height))
        merged_image.paste(images[3], (0 * width, 3 * height))
        merged_image.paste(images[4], (1 * width, 0 * height))
        merged_image.paste(images[5], (1 * width, 1 * height))
        merged_image.paste(images[6], (1 * width, 2 * height))
        merged_image.paste(images[7], (1 * width, 3 * height))
        merged_image.paste(images[8], (2 * width, 0 * height))
        merged_image.paste(images[9], (2 * width, 1 * height))
        merged_image.paste(images[10], (2 * width, 2 * height))
        merged_image.paste(images[11], (2 * width, 3 * height))
        merged_image.paste(images[12], (3 * width, 0 * height))
        merged_image.paste(images[13], (3 * width, 1 * height))
        merged_image.paste(images[14], (3 * width, 2 * height))
        merged_image.paste(images[15], (3 * width, 3 * height))
    if VERSION == 5:
        merged_image.paste(images[0], (0 * width, 0 * height))
        merged_image.paste(images[1], (0 * width, 1 * height))
        merged_image.paste(images[2], (0 * width, 2 * height))
        merged_image.paste(images[3], (0 * width, 3 * height))
        merged_image.paste(images[4], (0 * width, 4 * height))
        merged_image.paste(images[5], (1 * width, 0 * height))
        merged_image.paste(images[6], (1 * width, 1 * height))
        merged_image.paste(images[7], (1 * width, 2 * height))
        merged_image.paste(images[8], (1 * width, 3 * height))
        merged_image.paste(images[9], (1 * width, 4 * height))
        merged_image.paste(images[10], (2 * width, 0 * height))
        merged_image.paste(images[11], (2 * width, 1 * height))
        merged_image.paste(images[12], (2 * width, 2 * height))
        merged_image.paste(images[13], (2 * width, 3 * height))
        merged_image.paste(images[14], (2 * width, 4 * height))
        merged_image.paste(images[15], (3 * width, 0 * height))
        merged_image.paste(images[16], (3 * width, 1 * height))
        merged_image.paste(images[17], (3 * width, 2 * height))
        merged_image.paste(images[18], (3 * width, 3 * height))
        merged_image.paste(images[19], (3 * width, 4 * height))
        merged_image.paste(images[20], (4 * width, 0 * height))
        merged_image.paste(images[21], (4 * width, 1 * height))
        merged_image.paste(images[22], (4 * width, 2 * height))
        merged_image.paste(images[23], (4 * width, 3 * height))
        merged_image.paste(images[24], (4 * width, 4 * height))

    # Save the merged image
    output_path = os.path.join(output_folder, output_name)
    merged_image.save(output_path)

    for img in images:
        img.close()
n = 0
bad = 0
for i in tqdm(range(len(all_names))):
    n += 1
    patient = all_names[i][0][0]
    class_ = all_names[i][0][1]
    folder_path = r"C:/STUDIA/data/archive/IDC_regular_ps50_idx5/" + str(patient) + "/" + str(class_)# Specify the folder containing the image parts
    image_names = all_names[i][1:]  # List of image part filenames
    output_folder = r"C:/STUDIA/data/" + f"large_{VERSION}"  # Specify the folder where you want to save the merged image
    output_name = f"{class_}_large-{VERSION}_{i}.jpg"  # Specify the filename for the merged image
    try:
        merge_images(folder_path, image_names, output_folder, output_name)
    except FileNotFoundError as e:
        bad += 1

print(bad, n)






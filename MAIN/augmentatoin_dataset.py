import os
import shutil

from keras.src.saving.saving_api import load_model
from tqdm import tqdm

from MAIN.test_model import test_model
from extender import Extender
from my_constans import DATA_PATH
from create_dataset import split_and_move_images
from augmentation import Augmentator


def do_augmentation(n=100):
    cur_dir = os.curdir
    os.chdir(DATA_PATH)
    # os.mkdir("augmentation")
    for i in tqdm(range(n)):
        not_augmented_folder_name = f"{i}_not_augmented"
        augmented_folder_name = f"{i}_augmented"
        not_augmented_folder_path = DATA_PATH / "augmentation" / not_augmented_folder_name
        augmented_folder_path = DATA_PATH / "augmentation" / augmented_folder_name
        split_and_move_images(not_augmented_folder_path, 1, 0, fcn=shutil.copy, source_folder=DATA_PATH / "ready")
        split_and_move_images(augmented_folder_path, 0, 0, fcn=shutil.copy, source_folder=not_augmented_folder_path)
        augmentator = Augmentator([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        augmentator.add_function(augmentator.mirror_image(type_=0), stack_key=1) #

        augmentator.add_function(augmentator.mirror_image(type_=1), stack_key=2)

        augmentator.add_function(augmentator.mirror_image(type_=2), stack_key=3)

        augmentator.add_function(augmentator.mirror_image(type_=1), stack_key=4)
        augmentator.add_function(augmentator.mirror_image(type_=2), stack_key=4) #
        augmentator.add_function(augmentator.mirror_image(type_=0), stack_key=5)
        augmentator.add_function(augmentator.mirror_image(type_=1), stack_key=5) #

        augmentator.add_function(augmentator.mirror_image(type_=0), stack_key=6)
        augmentator.add_function(augmentator.mirror_image(type_=2), stack_key=6)

        augmentator.add_function(augmentator.add_gaussian_noise(), stack_key=7) #

        augmentator.add_function(augmentator.add_salt_and_pepper_noise(salt_prob=0.05, pepper_prob=0.05), stack_key=8) #

        augmentator.add_function(augmentator.stretch_image(factor_width=1.1, factor_height=1.1), stack_key=9)

        augmentator.add_function(augmentator.stretch_image(factor_width=0.9, factor_height=0.9), stack_key=10) #

        augmentator.add_function(augmentator.remove_fragment(width_factor=0.1, height_factor=0.1, n=10), stack_key=11)

        augmentator.add_function(augmentator.remove_fragment(width_factor=0, height_factor=0, n=0), stack_key=12)
        extender_0 = Extender(not_augmented_folder_path / "test" / "0", augmented_folder_path / "test" / "0", augmentator)
        extender_1 = Extender(not_augmented_folder_path / "test" / "1", augmented_folder_path / "test" / "1", augmentator)
        extender_0.extend_class("0", 13)
        extender_1.extend_class("1", 13)
    os.chdir(cur_dir)


def test_augmentation(model, n_images=100, n_channels=12):
    improvements = [
        {
            "keep_ok": 0,
            "keep_bad": 0,
            "improve": 0,
            "worsen": 0,
            "mean_output": 0
        } for _ in range(n_channels)]
    base_accuracies = []
    augmented_accuracies_sum = []
    augmented_accuracies_voting = []
    for i in tqdm(range(n_images)):
        augmented_folder_name = f"{i}_augmented"
        augmented_folder_path = DATA_PATH / "augmentation" / augmented_folder_name
        result, images = test_model(model, model_name, augmented_folder_path / "test", model=model, ret_images=True)
        names = images.filenames

        summary_2 = list(zip(result["summary"], names))
        acc_sum = 0
        acc_sum += 1 if sum(map(lambda x: x[2] if x[0] == 0 else 0, result["summary"]))/n_channels < 0.5 else 0
        acc_sum += 1 if sum(map(lambda x: x[2] if x[0] == 1 else 0, result["summary"]))/n_channels >= 0.5 else 0
        acc_sum /= 2
        augmented_accuracies_sum.append(acc_sum)

        acc_voting = 0
        acc_voting += 1 if sum(map(lambda x: int(x[1] == x[0]) if x[0] == 0 else 0, result["summary"])) / n_channels > 0.5 else 0
        acc_voting += 1 if sum(map(lambda x: int(x[1] == x[0]) if x[0] == 1 else 0, result["summary"])) / n_channels > 0.5 else 0
        acc_voting /= 2
        augmented_accuracies_voting.append(acc_voting)

        base_acc = 0
        good_predictions = [0, 0]
        for (true, pred, val), name in summary_2:
            method = int(name.split('.')[0].split('-')[-1])-1
            if method + 1 == n_channels:
                base_acc += int(true == pred)
                good_predictions[true] = (true == pred)
        base_accuracies.append(base_acc/2)
        for (true, pred, val), name in summary_2:
            method = int(name.split('.')[0].split('-')[-1])-1
            if method + 1 != n_channels:
                if good_predictions[true]:
                    if pred == true:
                        improvements[method]["keep_ok"] += 1
                    else:
                        improvements[method]["worsen"] += 1
                else:
                    if pred == true:
                        improvements[method]["improve"] += 1
                    else:
                        improvements[method]["keep_bad"] += 1

    return base_accuracies, augmented_accuracies_voting, augmented_accuracies_sum, improvements

if __name__ == "__main__":
    n_images = 200
    n_channels = 12
    # do_augmentation(n_images)
    model_name = "MobileNetV2_max_pull_93.keras"
    model = load_model(f"C:\STUDIA\Magisterka\MAIN\models\MobileNetV2_max_pull_93.keras")
    b_a, v_a, s_a, imp = test_augmentation(model, n_images, n_channels)
    print("Base:", sum(b_a)/len(b_a))
    print("Voiting:", sum(v_a)/len(v_a))
    print("Sum:", sum(s_a)/len(s_a))
    for i in range(n_channels):
        print(i + 1)
        print(imp[i])




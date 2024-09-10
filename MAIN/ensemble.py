#%%
import sys
from pathlib import Path

sys.path.extend(['C:\\STUDIA\\Magisterka\\MAIN'])
import os

from docutils.nodes import option_string
from keras.src.saving.saving_api import load_model
from sklearn.metrics import accuracy_score, recall_score
import pathlib

from MAIN.loaded_models import models
from MAIN.my_constans import V_SMALL_DATA_TEST, DATA_PATH
from MAIN.test_model import get_test_images, _test_model

models_names = list(os.listdir(r"D:\Magisterka\MAIN\models\Trained\original"))
print(models_names)
# models_names = [
#     "VGG16_best_92_from_0.keras",
#     "MobileNetV2_92.keras",
#     "ResNet50V2_93.keras",
#     "MobileNetV2_max_pull_93.keras",
#     "MobileNetV2_max_pull_92.keras",
#     "DenseNet169_92.keras",
#     # "EfficientNetV2M_92.keras"
#     ]
# models_names = [
# "MobileNetV20_multiple_5000.keras",
# "MobileNetV21_multiple_5000.keras",
# "MobileNetV22_multiple_5000.keras",
# "MobileNetV23_multiple_5000.keras",
# "MobileNetV24_multiple_5000.keras",
# "MobileNetV25_multiple_5000.keras",
# "MobileNetV26_multiple_5000.keras",
# "MobileNetV27_multiple_5000.keras",
# "MobileNetV28_multiple_5000.keras",
# "MobileNetV29_multiple_5000.keras",
# "MobileNetV210_multiple_5000.keras",
# "MobileNetV211_multiple_5000.keras",
# "MobileNetV212_multiple_5000.keras",
# "MobileNetV213_multiple_5000.keras",
# "MobileNetV214_multiple_5000.keras",
# "MobileNetV215_multiple_5000.keras",
# ]

def load_models(names):
    models_list = []
    functions = []
    models_dir = Path(r"D:\Magisterka\MAIN\models\Trained\original")
    print(models_dir)
    os.chdir(models_dir)
    for name in names:
        print(f"loading model {name}...")
        model = load_model(name)
        print(f"model {name} loaded")
        models_list.append(model)
        model_prefix = name.split("_")[0]
        func = models[model_prefix]["func"]
        functions.append(func)
    os.chdir(os.path.abspath(os.getcwd()))
    return zip(models_list, functions)

def predict_models(test_path, models_and_functions):
    summaries = []
    for model, function in models_and_functions:
        generator = get_test_images(test_path, function)
        result = _test_model(images=generator, model=model)
        summaries.append(result["summary"])
    summaries = list(zip(*summaries))
    return summaries

def process_results(results):
    true, predicted, predictions = list(zip(*results))
    true = true[0]
    return true, predicted, predictions

def voting(true, predicted, _):
    val = sum(predicted) / len(predicted)
    if val >= 0.5:
        predicted = 1
    else:
        predicted = 0
    return true[0], predicted, val

def sure_sum(true, _, predictions):
    val = sum(predictions) / (len(predictions))
    if sum(predictions) / (len(predictions)) > 0.5:
        predicted = 1
    else:
        predicted = 0
    return true[0], predicted, val


def most_sure(true, _, predictions):
    val = max(predictions) if 1-max(predictions) < min(predictions) else min(predictions)
    if 1-max(predictions) > min(predictions):
        predicted = 0
    else:
        predicted = 1
    return true[0], predicted, val


def process_results(results, fcn, ret_val=False):
    tr_pr = []
    for result in results:
        true, predicted, val = fcn(*list(zip(*result)))
        if ret_val:
            tr_pr.append((true, predicted, val))
        else:
            tr_pr.append((true, predicted))
    return list(zip(*tr_pr))


def calc_metrics(true, pred):
    acc = accuracy_score(true, pred)
    rec = recall_score(true, pred)
    print(f"Acc: {acc} \nRec: {rec}")


if __name__ == '__main__':
    #%%
    models_fcn = load_models(models_names)
    summaries = predict_models(r"D:\data\original\test", models_fcn)
    #%%
    true, predicted, val = process_results(summaries, voting, ret_val=True)
    for t, p, v in zip(true, predicted, val):
        if t != p:
            print(v)
    calc_metrics(true, predicted)
    #%%
    print(summaries)
    #%%
    bad_images = []
    def f(x):
        return x
    images = get_test_images(r"D:\data\original\test", f)
    sum_0 = 0
    sum_1 = 0
    n_0 = 0
    n_1 = 0
    _, predicted, _ = process_results(summaries, voting, ret_val=True)
    for i, pred, img in zip(summaries, predicted, images.filenames):
        t, p, v = list(zip(*i))
        t = t[0]
        if pred != t:
            print(t, p, v, img)
            bad_images.append(img)
            if t == 1:
                n_1 += 1
                sum_1 += sum(p)/6
            else:
                sum_0 += 1 - sum(p)/6
                n_0 += 1
    print(sum_0/n_0)
    print(sum_1/n_1)
    #%%
    print(bad_images)







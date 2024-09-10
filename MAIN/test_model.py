from keras.src.saving.saving_api import load_model
from sklearn.metrics import accuracy_score, recall_score

from my_constans import DATA_PATH
from my_utils import get_image_generator
from loaded_models import models

def test_model(model_path, model_name, dataset_path, images=None, model=None, ret_images=False):
    if model is None:
        model = load_model(model_path)
    if images is None:
        model_base_name = model_name.split('_')[0]
        preproces_fcn = models[model_base_name]["func"]
        images = get_test_images(dataset_path, preproces_fcn)
    result = _test_model(model, images)
    if ret_images:
        return result, images
    else:
        return result

def get_test_images(path, preproces_fcn):
    images = get_image_generator(path, preproces_fcn, test=True,
                                 batch_size=64,
                                 class_mode='binary', size=(150, 150))
    return images

def _test_model(model, images):
    true_labels = images.classes
    predictions = model.predict(images).tolist()
    predictions = list(map(lambda x: round(x[0], 3), predictions))
    predicted_labels = list(map(lambda x: int(x >= 0.5), predictions))
    summary = list(zip(true_labels, predicted_labels, predictions))
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    return {
        "summary": summary,
        "accuracy": accuracy,
        "recall": recall
    }

if __name__ == "__main__":
    model_name = r"C:\STUDIA\Magisterka\MAIN\models\MobileNetV2_max_pull_93.keras"
    data_path = DATA_PATH / "large" / "l_3_ready" / 'test'
    result = test_model(model_name, "MobileNetV2_max_pull_93.keras", data_path)
    print(result["accuracy"])
    print(result["recall"])

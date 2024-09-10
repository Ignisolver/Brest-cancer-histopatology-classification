#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.applications import ResNet50
from keras.src.saving.saving_api import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

def get_img_array(img_path, size):
    # Ładowanie obrazu
    img = image.load_img(img_path, target_size=size)
    # Konwertowanie obrazu do tablicy numpy
    img_array = image.img_to_array(img)
    # Rozszerzanie wymiarów obrazu, aby pasował do wymagań modelu ResNet50
    img_array = np.expand_dims(img_array, axis=0)
    # Normalizacja obrazu
    img_array = preprocess_input(img_array)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # Tworzenie nowego modelu, który zwraca aktywacje ostatniej warstwy konwolucyjnej i predykcje
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)
    with tf.GradientTape() as tape:
        # Pobieranie aktywacji warstwy ostatniej konwolucji dla danego obrazu
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Predykcje dla danego obrazu
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Obliczanie gradientów predykcji najwyższej klasy w stosunku do aktywacji warstwy konwolucyjnej
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Obliczanie średniej ważonej gradientów
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizacja heatmapy
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def show_gradcam(img_path, heatmap):
    # Ładowanie obrazu
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    # Nakładanie heatmapy na obraz
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Konwersja heatmapy do obrazu RGB
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Łączenie heatmapy z obrazem wejściowym
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = image.array_to_img(superimposed_img)

    # Wyświetlanie obrazu z heatmapą
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
#%%
# Ładowanie modelu ResNet50
# model = load_model(r"C:\STUDIA\Magisterka\MAIN\models\ResNet50V2_93.keras")
model = ResNet50()
#%%
# Wymiary obrazu
img_size = (224, 224)
# Ścieżka do obrazu
img_path = r"C:\STUDIA\data\ready_old\train\1\1_1_497-1.png"
# Nazwa warstwy konwolucyjnej, której aktywacje chcemy wykorzystać do generowania heatmapy
last_conv_layer_name = 'conv5_block3_out'

# Nazwy warstw klasyfikatora (w przypadku ResNet50, warstwy Dense po ostatniej warstwie konwolucyjnej)
classifier_layer_names = [
    'predictions'
]
# for layer in model.layers:
#     l = layer.get_layer(last_conv_layer_name)
#     print(layer)
# Przygotowanie obrazu
img_array = get_img_array(img_path, size=img_size)

# Generowanie heatmapy Grad-CAM++
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
#%%
# Wyświetlenie obrazu z heatmapą
show_gradcam(img_path, heatmap)

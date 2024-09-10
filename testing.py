from pathlib import Path

from sklearn.metrics import accuracy_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
import numpy as np
import os


model = load_model('./models/best_model_original_87.h5')

test_folder_1 = Path('../data/ready/0')
image_files_1 = list(np.random.choice(os.listdir(test_folder_1), 20, replace=False))
for i in range(len(image_files_1)):
    image_files_1[i] = test_folder_1.joinpath(image_files_1[i])

test_folder_2 = Path('../data/ready/1')
image_files = list(np.random.choice(os.listdir(test_folder_2), 20, replace=False))
for i in range(len(image_files)):
    image_files[i] = test_folder_2.joinpath(image_files[i])

image_files.extend(image_files_1)

true_labels = []
predicted_labels = []

for img_path in image_files:
    img = image.load_img(img_path, target_size=(50, 50))
    img_array = image.img_to_array(img)
    # img_array = preprocess_input(img_array)  # EfficientNetB0 requires specific preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    print(predictions)
    predicted_label = np.argmax(predictions)

    true_label = int(img_path.name.split('_')[0])  # Modify based on your filename format

    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print(list(zip(true_labels, predicted_labels)))

true_labels_onehot = to_categorical(true_labels, num_classes=2)

accuracy = accuracy_score(np.argmax(true_labels_onehot, axis=1), predicted_labels)
print(f'Categorical Accuracy: {accuracy:.4f}')

recall = recall_score(true_labels, predicted_labels, average='weighted')
print(f'Recall: {recall:.4f}')

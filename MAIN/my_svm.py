import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_label in os.listdir(folder):
        class_folder = os.path.join(folder, class_label)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    images.append(img)
                    labels.append(int(class_label))
    return images, labels

# Load images from folders
folder_path = "../../data/ready_2/train"
images, labels = load_images_from_folder(folder_path)

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.01, random_state=42)

# Flatten and normalize images
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
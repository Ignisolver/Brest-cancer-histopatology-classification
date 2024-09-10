import tensorflow as tf

# Wczytanie obrazu
image_path = 'data.png'
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image)

# Zmiana rozmiaru obrazu do wymaganego rozmiaru (224x224 pikseli)
resized_image = tf.image.resize(image, [224, 224])

# Normalizacja wartości pikseli do zakresu [0, 1]
normalized_image = resized_image / 255.0

# Dodanie dodatkowego wymiaru do obrazu (MobileNetV2 wymaga tensora 4D jako wejścia)
input_image = tf.expand_dims(normalized_image, axis=0)

# Załadowanie modelu MobileNetV2
model = tf.keras.applications.MobileNetV2()

# Klasyfikacja obrazu
predictions = model.predict(input_image)
import tensorflow as tf
from keras.src.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from MAIN.my_constans import DATA_PATH

# Define constants
IMAGE_SIZE = (50, 50)
BATCH_SIZE = 32

# Data directories
train_dir = DATA_PATH / "ready" / "train"

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.002
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(50, 50, 3))

# Add custom classification head
x = base_model.output
x = GlobalMaxPooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification with sigmoid activation

# Combine base model with custom head
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=100,  # Adjust as needed
    validation_split=0.002,
    validation_data=validation_generator,
)

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# Definiuj ścieżki do danych treningowych, walidacyjnych i testowych
dataset_dir = '../data/ready'
models_dir = Path('./models')

# Parametry modelu
img_size = (50, 50)
batch_size = 64
epochs = 17

# Przygotuj dane treningowe, walidacyjne i testowe za pomocą ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation'
)


# Zbuduj model EfficientNetB0
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
model = load_model(models_dir / "original.h5")
# model = models.Sequential()
# model.add(base_model)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dense(train_generator.num_classes, activation='sigmoid'))


def schedule(epoch, lr):
    lr = lr * (1-(epoch * 0.04))
    return lr

lr_scheduler = LearningRateScheduler(schedule)

# Kompiluj model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Definiuj callback do zapisywania wag modelu
checkpoint_filepath = 'best_model_original.h5'
model_checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

callbacks = [
    early_stopping,
    model_checkpoint,
    # lr_scheduler
]
# Trenuj model z early stopping i zapisywaniem wag
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)
## ------------

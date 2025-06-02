import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 1. Parametry Treningu
DATASET_ROOT_DIR = 'dataset_root'
IMAGE_SIZE = (160, 160) # Rozmiar obrazu dla MobileNetV2
BATCH_SIZE = 32
NUM_CLASSES = 9
EPOCHS = 20 # Sugerowana liczba epok do początkowego treningu

CLASS_NAMES = sorted(['11476', '15068', '24246', '2654', '35480', '4032', '60474', '63868', '85861'])

# 2. Przygotowanie Danych (Generatory Danych)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_ROOT_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_ROOT_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    subset='validation'
)

print(f"Znaleziono {train_generator.samples} obrazów treningowych ({train_generator.num_classes} klas).")
print(f"Znaleziono {validation_generator.samples} obrazów walidacyjnych ({validation_generator.num_classes} klas).")
print(f"Mapowanie klas: {train_generator.class_indices}")

if train_generator.num_classes != NUM_CLASSES or validation_generator.num_classes != NUM_CLASSES:
    print(f"BŁĄD: Niezgodność liczby klas! Oczekiwano {NUM_CLASSES}.")
    exit()

# 3. Budowa Modelu (Transfer Learning z MobileNetV2)
base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                         include_top=False, # Bez górnej warstwy klasyfikującej
                         weights='imagenet') # Wagi pre-trenowane na ImageNet

base_model.trainable = False # Zamrożenie wag modelu bazowego

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3), # Regularyzacja
    Dense(NUM_CLASSES, activation='softmax') # Warstwa wyjściowa
])

# 4. Kompilacja Modelu
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Trening Modelu
if train_generator.samples == 0 or validation_generator.samples == 0:
    print("BŁĄD: Brak obrazów w generatorach danych.")
else:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # 6. Wizualizacja Krzywych Uczenia
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność Treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
    plt.legend(loc='lower right')
    plt.title('Dokładność Treningowa i Walidacyjna')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata Treningowa')
    plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
    plt.legend(loc='upper right')
    plt.title('Strata Treningowa i Walidacyjna')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.savefig('training_history_mobilenetv2.png') # Zapis wykresu
    plt.show()

    # 7. Zapisanie Modelu
    model.save('lego_classifier_mobilenetv2.h5')
    print("Model został zapisany jako: lego_classifier_mobilenetv2.h5")
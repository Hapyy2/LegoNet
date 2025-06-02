import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 1. Parametry Dotrenowywania
DATASET_ROOT_DIR = 'dataset_root'
IMAGE_SIZE = (160, 160) # Musi być takie samo jak w pierwotnym treningu
BATCH_SIZE = 32
NUM_CLASSES = 9 # Musi być takie samo
ADDITIONAL_EPOCHS = 15 # Liczba dodatkowych epok
INITIAL_EPOCHS_DONE = 20 # Ile epok model już przeszedł (z poprzedniego treningu)

SAVED_MODEL_PATH = 'lego_classifier_mobilenetv2.h5' # Model do wczytania
CONTINUED_MODEL_SAVE_PATH = 'lego_classifier_mobilenetv2_continued.h5' # Nazwa dla dotrenowanego modelu

CLASS_NAMES = sorted(['11476', '15068', '24246', '2654', '35480', '4032', '60474', '63868', '85861'])

# 2. Przygotowanie Danych (identycznie jak w treningu)
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

if train_generator.num_classes != NUM_CLASSES or validation_generator.num_classes != NUM_CLASSES:
    print(f"BŁĄD: Niezgodność liczby klas! Oczekiwano {NUM_CLASSES}.")
    exit()
    
# 3. Wczytanie Zapisanego Modelu
print(f"Wczytywanie modelu z: {SAVED_MODEL_PATH}")
try:
    model = load_model(SAVED_MODEL_PATH)
    print("Model wczytany pomyślnie.")
except Exception as e:
    print(f"Błąd podczas wczytywania modelu: {e}")
    exit()

# 4. Kompilacja Modelu (zalecane użycie niższego learning rate)
NEW_LEARNING_RATE = 0.0001 # Dla fine-tuningu/dotrenowywania
model.compile(optimizer=Adam(learning_rate=NEW_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(f"Model skompilowany z nowym learning rate: {NEW_LEARNING_RATE}")

# 5. Dotrenowywanie Modelu
print(f"Rozpoczynanie dotrenowywania na kolejne {ADDITIONAL_EPOCHS} epok...")

if train_generator.samples == 0 or validation_generator.samples == 0:
    print("BŁĄD: Brak obrazów w generatorach danych.")
else:
    history_continued = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=INITIAL_EPOCHS_DONE + ADDITIONAL_EPOCHS,
        initial_epoch=INITIAL_EPOCHS_DONE # Kontynuacja numeracji epok
    )

    # 6. Wizualizacja Krzywych Uczenia po Dotrenowaniu
    acc = history_continued.history['accuracy']
    val_acc = history_continued.history['val_accuracy']
    loss = history_continued.history['loss']
    val_loss = history_continued.history['val_loss']

    epochs_range_continued = range(INITIAL_EPOCHS_DONE, INITIAL_EPOCHS_DONE + ADDITIONAL_EPOCHS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range_continued, acc, label='Dokładność Treningowa (dotren.)')
    plt.plot(epochs_range_continued, val_acc, label='Dokładność Walidacyjna (dotren.)')
    plt.legend(loc='lower right')
    plt.title('Dokładność po Dotrenowaniu')
    plt.xlabel(f'Epoka (kontynuacja od {INITIAL_EPOCHS_DONE})')
    plt.ylabel('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range_continued, loss, label='Strata Treningowa (dotren.)')
    plt.plot(epochs_range_continued, val_loss, label='Strata Walidacyjna (dotren.)')
    plt.legend(loc='upper right')
    plt.title('Strata po Dotrenowaniu')
    plt.xlabel(f'Epoka (kontynuacja od {INITIAL_EPOCHS_DONE})')
    plt.ylabel('Strata')
    plt.savefig('continued_training_history_mobilenetv2.png') # Zapis wykresu
    plt.show()

    # 7. Zapisanie Dotrenowanego Modelu
    model.save(CONTINUED_MODEL_SAVE_PATH)
    print(f"Dotrenowany model został zapisany jako: {CONTINUED_MODEL_SAVE_PATH}")
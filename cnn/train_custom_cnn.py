import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# --- Parametry ---
DATA_DIR = r'C:\Users\mateu\Desktop\Inteligencja\KlockiLego\dataset_root' # Ścieżka do głównego folderu z danymi (podfoldery to klasy)
MODEL_SAVE_PATH = 'lego_brick_classifier_model.keras'

SELECTED_BRICK_CODES = sorted(['11476', '15068', '24246', '2654', '35480', '4032', '60474', '63868', '85861'])
NUM_CLASSES = len(SELECTED_BRICK_CODES)
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25 # Można dostosować

# --- Wczytywanie Danych ---
if not os.path.isdir(DATA_DIR):
    print(f"BŁĄD: Folder danych nie istnieje: {DATA_DIR}")
    exit()

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=SELECTED_BRICK_CODES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=SELECTED_BRICK_CODES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Załadowane klasy (kolejność ważna): {train_dataset.class_names}")
if train_dataset.class_names != SELECTED_BRICK_CODES:
    print("OSTRZEŻENIE: Niezgodność kolejności klas!")

# --- Augmentacja Danych (jako warstwa modelu) ---
data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.2),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomContrast(factor=0.2),
    ],
    name="data_augmentation"
)

# Optymalizacja wczytywania
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# --- Definicja Modelu CNN ---
input_shape_model = (IMG_SIZE[0], IMG_SIZE[1], 3)
model = keras.Sequential([
    layers.Input(shape=input_shape_model),
    layers.Rescaling(1./255), # Normalizacja
    data_augmentation,         # Augmentacja
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Regularyzacja
    layers.Dense(NUM_CLASSES, activation='softmax') # Warstwa wyjściowa
])

# --- Kompilacja Modelu ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Trening Modelu ---
print("Rozpoczynanie treningu modelu...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]
)

# --- Ewaluacja i Wizualizacja ---
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"\nStrata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_accuracy:.4f}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss_hist = history.history['loss']
val_loss_hist = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Dokładność Treningowa')
plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
plt.legend(loc='lower right')
plt.title('Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_hist, label='Strata Treningowa')
plt.plot(epochs_range, val_loss_hist, label='Strata Walidacyjna')
plt.legend(loc='upper right')
plt.title('Strata')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.savefig('custom_cnn_training_history.png')
plt.show()

# --- Zapis Modelu ---
model.save(MODEL_SAVE_PATH)
print(f"Model zapisany jako: {MODEL_SAVE_PATH}")
print(f"Kolejność klas w modelu: {train_dataset.class_names}")
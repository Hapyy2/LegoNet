import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- Główna Funkcja Ewaluacyjna ---
def evaluate_specific_model(model_path, model_name, test_data_dir, target_image_size, class_names_list, needs_manual_rescale):
    print(f"\n--- Ewaluacja modelu: {model_name} ---")
    print(f"Ścieżka: {model_path} | Rozmiar obrazu: {target_image_size}")
    if needs_manual_rescale:
        print("Info: Dla tego modelu zostanie zastosowane ręczne skalowanie danych wejściowych do [0,1].")

    if not os.path.exists(model_path):
        print(f"BŁĄD: Plik modelu nie znaleziony: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Błąd wczytywania modelu {model_path}: {e}")
        return None

    if not os.path.isdir(test_data_dir):
        print(f"BŁĄD: Folder testowy nie znaleziony: {test_data_dir}")
        return None

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names_list,
        image_size=target_image_size,
        shuffle=False, # Kluczowe dla spójnej ewaluacji
        batch_size=32,
        interpolation='nearest'
    )

    if test_dataset.class_names != class_names_list:
        print("BŁĄD KRYTYCZNY: Niezgodność nazw klas z folderu testowego i zdefiniowanych.")
        return None
        
    y_true_categorical_list = []
    y_pred_probabilities_list = []

    for images_batch, labels_batch in test_dataset:
        current_images_batch = images_batch
        if needs_manual_rescale:
            # Ręczne skalowanie dla modeli, które tego wymagają (np. MobileNetV2 bez wbudowanej warstwy Rescaling)
            current_images_batch = tf.cast(images_batch, tf.float32) / 255.0 
        
        y_true_categorical_list.extend(labels_batch.numpy())
        batch_predictions = model.predict_on_batch(current_images_batch)
        y_pred_probabilities_list.extend(batch_predictions)

    y_true_cat_np = np.array(y_true_categorical_list)
    y_pred_probs_np = np.array(y_pred_probabilities_list)

    y_true_indices = np.argmax(y_true_cat_np, axis=1)
    y_pred_indices = np.argmax(y_pred_probs_np, axis=1)

    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names_list, digits=3, zero_division=0)
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    print(f"\n## Raport dla: {model_name} ##")
    print(f"Dokładność: {accuracy*100:.2f}%\n")
    print("Raport Klasyfikacji:")
    print(report)
    print("\nMacierz Pomyłek:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.title(f'Macierz Pomyłek - {model_name}')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png') # Zapis macierzy do pliku
    plt.show()
    
    return {"accuracy": accuracy, "report": report, "cm": cm, "model_name": model_name}

# --- Główna Część Skryptu Porównawczego ---
if __name__ == "__main__":
    MAIN_TEST_DATA_DIR = 'dataset_test' # Folder ze zdjęciami testowymi
    
    # Nazwy klas (muszą być spójne i odpowiadać nazwom folderów)
    MAIN_CLASS_NAMES = sorted(['11476', '15068', '24246', '2654', '35480', '4032', '60474', '63868', '85861'])

    # Definicje modeli do porównania
    models_to_compare_config = [
        {
            "model_path": "lego_brick_classifier_model.keras", # Model własny
            "model_name": "Model Własny (64x64)",
            "image_size": (64, 64),
            "needs_manual_rescale": False # Zakładamy, że ten model ma wbudowaną warstwę Rescaling
        },
        {
            "model_path": "lego_classifier_mobilenetv2_continued.h5", # Model MobileNetV2
            "model_name": "Model MobileNetV2 (160x160)",
            "image_size": (160, 160),
            "needs_manual_rescale": True # Wymaga ręcznego skalowania, jeśli nie ma warstwy Rescaling, a był trenowany na danych [0,1]
        }
    ]

    all_model_evaluation_results = []
    for model_config in models_to_compare_config:
        single_model_results = evaluate_specific_model(
            model_config["model_path"],
            model_config["model_name"],
            MAIN_TEST_DATA_DIR,
            model_config["image_size"],
            MAIN_CLASS_NAMES,
            model_config["needs_manual_rescale"]
        )
        if single_model_results:
            all_model_evaluation_results.append(single_model_results)

    # Podsumowanie Porównawcze
    print("\n\n=== PODSUMOWANIE PORÓWNAWCZE MODELI ===")
    if not all_model_evaluation_results:
        print("Nie udało się uzyskać wyników ewaluacji dla żadnego z modeli.")
    for res in all_model_evaluation_results:
        print(f"\nModel: {res['model_name']}")
        print(f"  Dokładność: {res['accuracy']*100:.2f}%")
        print("  Kluczowe metryki (uśrednione makro z raportu klasyfikacji):")
        try:
            report_lines = res['report'].split('\n')
            parsed_metrics = False
            for line in report_lines:
                if 'macro avg' in line.strip():
                    parts = line.split()
                    metric_values = [p for p in parts if p.replace('.', '', 1).isdigit()] # Prosta heurystyka do znalezienia liczb
                    if len(metric_values) >= 3:
                        print(f"    Precyzja (macro avg): {metric_values[0]}")
                        print(f"    Czułość (macro avg):  {metric_values[1]}")
                        print(f"    F1-score (macro avg): {metric_values[2]}")
                        parsed_metrics = True
                    else:
                        print("    Nie znaleziono wystarczających wartości liczbowych dla metryk 'macro avg'.")
                    break
            if not parsed_metrics:
                 print("    Nie znaleziono linii 'macro avg' w raporcie klasyfikacji lub nie udało się sparsować metryk.")
        except Exception as e:
            print(f"    Wystąpił błąd podczas parsowania szczegółowych metryk z raportu: {e}")

    print("\nPorównanie modeli zakończone.")
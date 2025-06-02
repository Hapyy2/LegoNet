# 🧱 LegoNet - Twój Asystent LEGO

Witaj w projekcie LegoNet! 🎉

## 🚀 O Projekcie

LegoNet to aplikacja (obecnie w fazie prototypu), która ma dwa główne cele:
1.  **(Identyfikacja) Klocków LEGO:** Prześlij zdjęcie klocka, a aplikacja spróbuje go rozpoznać! 🔍
2.  **Katalogowanie Zestawów LEGO:** W przyszłości LegoNet pomoże Ci sprawdzić, jakie zestawy możesz zbudować z posiadanych klocków. Na razie możesz prototypowo definiować zestawy. 🧩

## 🛠️ Użyte Technologie

* Python 🐍
* TensorFlow & Keras 🧠 (do budowy modeli AI)
* Streamlit 🎈 (do stworzenia tej prostej aplikacji webowej)
* Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (do analizy danych i wsparcia AI)

## 🌟 Kluczowe Elementy Projektu

1.  **Model Własny (LegoNet-CNN):** Nasz główny model do klasyfikacji klocków, osiągający świetne wyniki! 🏆
2.  **Porównanie Modeli:** Analiza i porównanie z modelem opartym na MobileNetV2.
3.  **Prototyp Aplikacji Webowej:** Interaktywna aplikacja Streamlit do testowania klasyfikatora i definiowania zestawów.

## 🚦 Jak Uruchomić Aplikację Lokalnie?

1.  **Klonuj repozytorium** .
2.  **Zainstaluj zależności:**
    ```bash
    pip install streamlit tensorflow Pillow numpy pandas scikit-learn matplotlib seaborn
    ```
3.  **Upewnij się, że masz model:** Plik `lego_brick_classifier_model.keras` powinien znajdować się w głównym katalogu projektu.
4.  **Uruchom aplikację Streamlit:**
    ```bash
    streamlit run app.py
    ```
5.  Otwórz przeglądarkę pod adresem `http://localhost:8501` 🌐

## 📁 Struktura Projektu (ważniejsze pliki)

* `app.py`: Główny plik aplikacji webowej Streamlit.
* `lego_brick_classifier_model.keras`: Wytrenowany model LegoNet-CNN.
* `train_custom_cnn.py`: Skrypt do trenowania modelu LegoNet-CNN.
* `run_comparison.py`: Skrypt do ewaluacji i porównania modeli.
* `LegoNet-1.pdf` (lub podobny): Pełny raport projektowy.
* `mobile_net/`: Folder ze skryptami i modelem dla MobileNetV2.
* `cnn/`: Folder z artefaktami dla modelu LegoNet-CNN (np. wykresy).
* `dataset_test/`: Folder z danymi testowymi.
* `dataset_root/`: Główny folder z danymi treningowymi/walidacyjnymi.

## 🧑‍💻 Autor

* Mateusz Klemann

Dzięki za sprawdzenie projektu! Miłej zabawy z klockami! 😊

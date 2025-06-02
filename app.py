import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import os

# --- Konfiguracja Aplikacji ---
MODEL_PATH = 'lego_brick_classifier_model.keras'
IMG_SIZE = (64, 64)
CLASS_NAMES = sorted(['11476', '15068', '24246', '2654', '35480', '4032', '60474', '63868', '85861'])


# --- Funkcje Pomocnicze ---
@st.cache_resource  # Cache'owanie modelu
def load_keras_model(path):
    if not os.path.exists(path):
        st.error(f"BŁĄD: Plik modelu '{path}' nie został znaleziony!")
        return None
    try:
        model = tf.keras.models.load_model(path)
        print(f"Model '{path}' wczytany.")
        return model
    except Exception as e:
        st.error(f"Błąd wczytywania modelu: {e}")
        return None


def preprocess_image(pil_image, target_size):
    img = pil_image.convert('RGB').resize(target_size)
    img_array = keras_image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)  # Model oczekuje batcha


def predict_brick_id(model_instance, pil_img, target_size, class_names_list):
    if model_instance is None:
        return "Błąd: Model niezaładowany", 0.0

    processed_img_tensor = preprocess_image(pil_img, target_size)
    try:
        predictions_probabilities = model_instance.predict(processed_img_tensor)
        predicted_class_idx = np.argmax(predictions_probabilities[0])
        predicted_label = class_names_list[predicted_class_idx]
        confidence_score = np.max(predictions_probabilities[0])
        return predicted_label, confidence_score
    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")
        return "Błąd predykcji", 0.0


# Wczytanie modelu
lego_model = load_keras_model(MODEL_PATH)

# --- Inicjalizacja Stanu Sesji Streamlit ---
if 'defined_lego_sets' not in st.session_state:
    st.session_state.defined_lego_sets = []
if 'current_set_name' not in st.session_state:
    st.session_state.current_set_name = ""
if 'current_set_bricks' not in st.session_state:
    st.session_state.current_set_bricks = []  # Lista słowników {'id': id_klocka, 'qty': ilosc}
if 'user_owned_bricks' not in st.session_state:
    st.session_state.user_owned_bricks = {}  # Słownik {id_klocka: ilosc}

# --- Interfejs Użytkownika ---
st.sidebar.title("LegoNet 🧱")
app_module_selection = st.sidebar.selectbox(
    "Wybierz moduł aplikacji:",
    ["Klasyfikator Klocków LEGO", "Definiowanie Zestawu LEGO (Prototyp)"]
)

if app_module_selection == "Klasyfikator Klocków LEGO":
    st.title("🧐 Klasyfikator Klocków LEGO")
    st.write("Prześlij zdjęcie klocka, aby model spróbował go zidentyfikować.")

    uploaded_image_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png"])

    if uploaded_image_file is not None:
        pil_image_instance = Image.open(uploaded_image_file)
        st.image(pil_image_instance, caption="Przesłane zdjęcie", use_column_width=True)

        if lego_model is None:
            st.error("Model klasyfikacyjny nie jest dostępny.")
        else:
            if st.button("🔎 Sklasyfikuj Klocek"):
                with st.spinner("Analizuję obraz..."):
                    predicted_brick_label, prediction_confidence = predict_brick_id(
                        lego_model, pil_image_instance, IMG_SIZE, CLASS_NAMES
                    )

                if "Błąd" not in predicted_brick_label:
                    st.success(f"**Przewidziany kod klocka:** {predicted_brick_label}")
                    st.info(f"**Pewność predykcji:** {prediction_confidence * 100:.2f}%")

                    # Prototyp: Dodanie do "posiadanych"
                    current_owned = st.session_state.user_owned_bricks.get(predicted_brick_label, 0)
                    st.session_state.user_owned_bricks[predicted_brick_label] = current_owned + 1
                    st.write(f"Dodano klocek '{predicted_brick_label}' do Twojej wirtualnej kolekcji.")
                else:
                    st.error("Nie udało się sklasyfikować obrazu.")
    else:
        st.info("Oczekuję na przesłanie zdjęcia klocka.")

elif app_module_selection == "Definiowanie Zestawu LEGO (Prototyp)":
    st.title("📝 Definiowanie Zestawu LEGO (Prototyp)")
    st.write("Wprowadź dane dla nowego zestawu. Dane są przechowywane tylko w tej sesji.")

    with st.form("new_lego_set_form", clear_on_submit=False):  # clear_on_submit=False dla lepszej kontroli stanu
        st.subheader("Nowy Zestaw LEGO")

        # Użycie kluczy do inputów, aby Streamlit nie resetował ich wartości niepotrzebnie
        set_name_val = st.text_input(
            "Nazwa zestawu LEGO:",
            value=st.session_state.current_set_name,
            key="set_name_input_field"
        )
        st.session_state.current_set_name = set_name_val  # Aktualizuj stan sesji na bieżąco

        st.write("**Potrzebne klocki do tego zestawu:**")
        if st.session_state.current_set_bricks:
            for i, brick_item in enumerate(st.session_state.current_set_bricks):
                st.text(f"  {i + 1}. ID: {brick_item['id']}, Ilość: {brick_item['qty']}")
        else:
            st.caption("  Nie dodano jeszcze żadnych klocków.")

        col1, col2, col3 = st.columns([2, 1, 1.2])
        with col1:
            brick_id_val = st.text_input("ID Klocka:", key="brick_id_field")
        with col2:
            brick_qty_val = st.number_input("Ilość:", min_value=1, step=1, value=1, key="brick_qty_field")
        with col3:
            st.write("")  # Placeholder dla wyrównania przycisku
            st.write("")
            add_brick_button_clicked = st.form_submit_button("➕ Dodaj Klocek")

        if add_brick_button_clicked:
            if brick_id_val and brick_qty_val > 0:
                st.session_state.current_set_bricks.append({'id': brick_id_val, 'qty': brick_qty_val})
                # Nie resetuj tu pól, użytkownik może chcieć dodać więcej podobnych
                st.success(f"Dodano klocek {brick_id_val} (ilość: {brick_qty_val}) do listy.")
                # st.rerun() # Rerun może być potrzebny, jeśli chcemy natychmiast odświeżyć listę bez pełnego submitu formularza
            else:
                st.warning("Podaj ID klocka i prawidłową ilość.")

        st.divider()
        define_set_button_clicked = st.form_submit_button("💾 Zdefiniuj Cały Zestaw")

        if define_set_button_clicked:
            if st.session_state.current_set_name and st.session_state.current_set_bricks:
                new_defined_set = {
                    "name": st.session_state.current_set_name,
                    "bricks": list(st.session_state.current_set_bricks)  # Utwórz kopię listy
                }
                st.session_state.defined_lego_sets.append(new_defined_set)
                st.success(f"Zestaw '{st.session_state.current_set_name}' został zdefiniowany!")

                # Resetowanie stanu dla nowego zestawu
                st.session_state.current_set_name = ""
                st.session_state.current_set_bricks = []
                # Wyczyść klucze inputów, aby formularz był pusty dla następnego zestawu
                st.session_state.set_name_input_field = ""
                st.session_state.brick_id_field = ""
                st.session_state.brick_qty_field = 1
                st.rerun()  # Odśwież, aby zobaczyć zmiany i wyczyścić formularz
            elif not st.session_state.current_set_name:
                st.warning("Proszę podać nazwę zestawu.")
            else:  # Brak klocków
                st.warning("Proszę dodać przynajmniej jeden klocek do zestawu.")

    st.divider()
    st.subheader("📋 Zdefiniowane Zestawy LEGO (w tej sesji):")
    if st.session_state.defined_lego_sets:
        for i, defined_set in enumerate(st.session_state.defined_lego_sets):
            with st.expander(f"{i + 1}. {defined_set['name']}"):
                for brick in defined_set['bricks']:
                    st.markdown(f"- Klocek ID: `{brick['id']}`, Ilość: **{brick['qty']}**")
    else:
        st.info("Nie zdefiniowano jeszcze żadnych zestawów LEGO w tej sesji.")

# Wyświetlanie "posiadanych" klocków
st.sidebar.divider()
st.sidebar.subheader("Moja Kolekcja (Prototyp)")
if st.session_state.user_owned_bricks:
    for brick_id_owned, count_owned in st.session_state.user_owned_bricks.items():
        st.sidebar.text(f"ID: {brick_id_owned} - Ilość: {count_owned}")
else:
    st.sidebar.caption("Twoja kolekcja jest pusta.")


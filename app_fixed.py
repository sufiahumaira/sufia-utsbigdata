import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is not None:
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        else:
            st.warning("Model YOLO belum siap digunakan.")

    elif menu == "Klasifikasi Gambar":
        if classifier is not None:
            img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            st.write("### Hasil Prediksi:", class_index)
            st.write("Probabilitas:", np.max(prediction))
        else:
            st.warning("Model klasifikasi belum siap digunakan.")

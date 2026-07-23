import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "major project" / "efficientnetb0.keras"
    return tf.keras.models.load_model(model_path)

model = load_model()

# -------------------------------
# Class Names
# IMPORTANT:
# Keep the order EXACTLY the same
# as during model training.
# -------------------------------
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-Damage",
    "Physical-Damage",
    "Snow"
]

# -------------------------------
# Title
# -------------------------------
st.title("☀️ Solar Panel Defect Classifier")

st.write(
    "Upload a solar panel image to detect defects using the trained EfficientNetB0 model."
)

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -------------------------------
    # Preprocessing
    # -------------------------------
    img = image.resize((224, 224))

    img_array = np.array(img).astype(np.float32)

    img_array = np.expand_dims(img_array, axis=0)
    

    # -------------------------------
    # Prediction
    # -------------------------------
    with st.spinner("Analyzing Solar Panel..."):

        predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    confidence = float(predictions[0][predicted_index])

    predicted_class = CLASSES[predicted_index]

    # -------------------------------
    # Result
    # -------------------------------
    st.success(f"### Prediction: {predicted_class}")

    st.write(f"**Confidence:** {confidence:.2%}")

    st.progress(confidence)

    if predicted_class == "Clean":
        st.success("✅ Solar panel is clean and healthy.")
    else:
        st.warning("⚠️ Defect/contamination detected.")

    # -------------------------------
    # Top 3 Predictions
    # -------------------------------
    st.subheader("Top 3 Predictions")

    top3 = np.argsort(predictions[0])[::-1][:3]

    for i, idx in enumerate(top3):

        st.write(
            f"**{i+1}. {CLASSES[idx]}** — {predictions[0][idx]:.2%}"
        )

    # -------------------------------
    # Probability Chart
    # -------------------------------
    st.subheader("Prediction Probabilities")

    df = pd.DataFrame({
        "Class": CLASSES,
        "Probability": predictions[0]
    })

    st.bar_chart(
        df.set_index("Class")
    )

    # -------------------------------
    # Detailed Probabilities
    # -------------------------------
    with st.expander("View Detailed Probabilities"):

        for i, prob in enumerate(predictions[0]):

            st.write(
                f"**{CLASSES[i]} :** {prob:.2%}"
            )
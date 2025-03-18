import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model (ensure you have a saved model in the same directory)
MODEL_PATH = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Example: 3 diseases)
CLASS_NAMES = ["Healthy", "Powdery Mildew", "Leaf Rust"]

# Pesticide recommendations based on disease
PESTICIDE_SUGGESTIONS = {
    "Healthy": "No pesticide needed. Maintain good plant care.",
    "Powdery Mildew": "Use Sulfur-based fungicides or Neem oil.",
    "Leaf Rust": "Apply Chlorothalonil or Mancozeb-based fungicides."
}

# Streamlit UI
st.title("Plant Disease Identification & Pesticide Suggestion")
st.write("Upload a leaf image to identify disease and get pesticide recommendations.")

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))  # Resize to match model input
    img_scaled = img_resized / 255.0  # Normalize
    img_expanded = np.expand_dims(img_scaled, axis=0)  # Expand dimensions

    # Predict using the model
    prediction = model.predict(img_expanded)
    predicted_class = np.argmax(prediction, axis=1)[0]
    disease_name = CLASS_NAMES[predicted_class]
    
    # Display results
    st.subheader(f"Disease Identified: **{disease_name}**")
    st.subheader(f"Pesticide Suggestion: {PESTICIDE_SUGGESTIONS[disease_name]}")

    # Debugging: Show confidence scores
    st.write("Confidence scores:", {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))})

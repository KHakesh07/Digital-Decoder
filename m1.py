import pickle
import streamlit as st
from PIL import Image
import numpy as np

# Function to load the model (called once outside the loop)
def load_model():
    with open('Digital_decoder.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model (executed only once)
model = load_model()

def preprocess_image(image_data):
    try:
        img = Image.open(image_data)
        print(f"Image format: {img.format}, mode: {img.mode}")  # Optional: Print format/mode

        # Check if grayscale (optional)
        if img.mode != 'L':
            img = img.convert('L')  # Convert to grayscale (optional)

        img = img.resize((28, 28), Image.LANCZOS)  # Use Lanczos resampling

        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values (optional)
        img_array = img_array.reshape(1, -1)  # Reshape for prediction
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None
    
def predict_digit(model, image_data):
    if image_data is not None:
        prediction = model.predict(image_data)
        return int(prediction[0])
    else:
        return None

# Simple UI with a file uploader and prediction display
uploaded_file = st.file_uploader("Upload an image:", type=None)  # Accept all types

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    if image is not None:
        prediction = predict_digit(model, image)
        if prediction is not None:
            st.write(f"Predicted digit: {prediction}")
        else:
            st.warning("Failed to make a prediction.")
else:
    st.info("Upload an image to make a prediction.")

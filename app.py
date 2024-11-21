import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
model = load_model('cat_dog_model.keras')

# Set background watermark using custom CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://i.pinimg.com/736x/d1/d1/81/d1d18156545ed373db49a08fa9896488.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            height: 100vh; /* Ensure the background image covers the whole screen */
        }
        .main .block-container {
            padding-top: 0rem;  /* Remove top padding to eliminate the white space */
            padding-bottom: 5rem;  /* Padding for footer */
            background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent background for text */
            border-radius: 15px;  /* Rounded corners for content */
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1, h2 {
            color: #333333; /* Darker text for visibility */
            font-family: 'Arial', sans-serif;
        }
        /* Styling the uploaded image with a border */
        .uploaded-image {
            border: 5px solid #333; /* Black border around the image */
            border-radius: 10px; /* Optional: rounded corners for the image */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        /* Styling the classification result */
        .result {
            background-color: rgba(0, 0, 0, 0.7); /* Dark background for better contrast */
            color: white; /* White text */
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            margin-top: 2rem;  /* Add margin for spacing */
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Cat😺 vs Dog🐶 Image Classification")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image with a border using HTML
    img = Image.open(uploaded_file)

    # Save the uploaded file temporarily as a byte stream and display
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Display image using Streamlit's st.image method
    st.image(img_bytes, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((64, 64))  # Resize image to match model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)

    # Display result with styling
    if prediction[0] < 0.5:
        st.markdown('<div class="result">The image is classified as: <strong>Cat 😺</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">The image is classified as: <strong>Dog 🐶</strong></div>', unsafe_allow_html=True)

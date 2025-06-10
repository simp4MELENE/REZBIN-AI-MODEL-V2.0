

import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import requests
from utils import PREPROCESS_TRANSFORM, load_model

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/MobileNetV2/mobilenetv2.pth")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gVVQVhOHKfA5S179M1E9VJnbMvzw9q_s"
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to download the model if it doesn't exist
def download_model_if_needed(model_path, url):
    if not model_path.exists():
        st.info("Downloading model file. Please wait...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model downloaded successfully.")
        else:
            st.error("Failed to download model from Google Drive.")
            st.stop()

def main():
    st.set_page_config(page_title="Rezbin AI Trash Classifier", page_icon="♻️")

    # Ensure model is downloaded
    download_model_if_needed(MODEL_PATH, MODEL_URL)

    # Load model
    model = load_model(MODEL_PATH, len(CLASS_LABELS), DEVICE)
    
    if model is None:
        st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the file exists.")
        return
    
    # UI
    st.image("./static/RezbinLogo.jpg", width=150)
    st.title("Rezbin AI Trash Classifier V2")
    st.markdown("""
        Upload an image of trash to classify it into one of the following categories:
        **Cardboard, Glass, Metal, Paper, Plastic, Trash**.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess
        img_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            model.eval()
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            prediction_label = CLASS_LABELS[predicted_idx.item()]
        
        prediction = prediction_label.upper()

        # Point system
        points = {
            "PLASTIC": 3,
            "CARDBOARD": 2,
            "METAL": 5,
            "PAPER": 1,
            "GLASS": 0,
            "TRASH": 0
        }
        point = points.get(prediction, 0)

        st.info(f"Image is: {prediction}. You earned {point} points.") 
        
        st.markdown("---")
        st.markdown("Developed by the Interns of Rezbin AI Engineers")

if __name__ == "__main__":
    main()


import streamlit as st
from PIL import Image
import torch
from pathlib import Path
from utils import PREPROCESS_TRANSFORM, load_model

# Set const values
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("./models/MobileNetV2/mobilenetv2.pth")
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def main():
    st.set_page_config(page_title="Rezbin AI Trash Classifier", page_icon="♻️")
    
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
    st.write("Upload an image of trash to classify it into one of the categories.")

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

        st.success(f"Predicted Class: **{prediction}**")
        
        # TODO: Create a logic that generates a variety of points based on what class is predicted
        sample_points = 0.5
        st.info(f"You generated {sample_points} points for predictiong {prediction}") 
        
        st.markdown("---")
        st.markdown("Developed by the Interns of Rezbin AI Engineers")
if __name__ == "__main__":
    main()
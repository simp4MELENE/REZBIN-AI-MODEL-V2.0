
import streamlit as st
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels from TrashNet
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_labels))
    checkpoint_path = os.path.join(os.path.dirname(__file__), "models", "01_mobilenetv2_finetuned.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

model = load_model()

# Define preprocessing pipeline (same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# UI
st.image("RezbinLogo.jpg", width=150)
st.title("Rezbin AI Trash Classifier")
st.write("Upload an image of trash to classify it into one of the categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((200, 200))  # Width x Height in pixels
    st.image(resized_image, caption='Uploaded Image', width=400)



    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        prediction = class_labels[predicted.item()]

    st.success(f"Predicted Class: **{prediction.upper()}**")
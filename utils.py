import streamlit as st
from torchvision import transforms, models
import torch
import torch.nn as nn
from pathlib import Path

PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model(model_path: Path, num_class_labels: int, device):
    model = model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_class_labels)
    
    if not model_path.exists():
        st.error(f"Error: Model file not found at {model_path}. Please update MODEL_PATH.")
        return None
    try:
        # Load the state_dict directly
        checkpoint = torch.load(model_path, map_location=device)
        # Check if the checkpoint is a full state_dict or a dict containing 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            model.load_state_dict(checkpoint) # Assume it's just the state_dict
        model.eval() # Set the model to evaluation mode
        model.to(device)
        return model

    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None
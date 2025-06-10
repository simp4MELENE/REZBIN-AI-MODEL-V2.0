import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

# Define the same preprocessing transform used during training
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes, device):
    """
    Load a MobileNetV2 model with the correct number of output classes.

    Args:
        model_path (Path): Path to the .pth model file.
        num_classes (int): Number of output classes.
        device (torch.device): CPU or GPU.

    Returns:
        model (torch.nn.Module): Loaded model ready for inference.
    """
    try:
        # Load a pre-trained MobileNetV2 model
        model = models.mobilenet_v2(pretrained=False)
        
        # Replace the classifier with one suitable for our number of classes
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        # Load the model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# List of skin disease labels
disease_labels = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatitis',
    'lichen planus',
    'melanoma',
    'psoriasis',
    'rosacea',
    'Seborrheic keratosis'
]

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        # Load the DINOv2 model
        try:
            self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f"Error loading DINOv2 from torch.hub: {e}")
            print("Attempting to load locally...")
            # Fallback to local load or placeholder
            from torchvision.models import vit_small_patch16_224
            self.transformer = vit_small_patch16_224(pretrained=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 8 skin disease classes
        )

    def forward(self, x):
        x = self.transformer(x)
        if hasattr(self.transformer, 'norm'):
            x = self.transformer.norm(x)
        return self.classifier(x)

def preprocess_image(image):
    """
    Preprocess an image for the DINO classifier.
    
    Args:
        image: PIL Image or path to image
        
    Returns:
        Preprocessed tensor
    """
    preprocess = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
    ])
    
    if isinstance(image, str):
        image = Image.open(image)
        
    return preprocess(image).unsqueeze(0)

def predict_skin_disease(model, image_path):
    """
    Predict skin disease from an image.
    
    Args:
        model: DINO classifier model
        image_path: Path to image file
        
    Returns:
        tuple: (probability, disease_name)
    """
    device = next(model.parameters()).device
    
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    input_tensor = preprocess_image(image).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()

    predicted_probability = probabilities[0, predicted_index].item()
    predicted_disease = disease_labels[predicted_index]

    return predicted_probability, predicted_disease

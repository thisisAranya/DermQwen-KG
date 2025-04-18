import re
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

def extract_response(text):
    """
    Extract assistant's response from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Extracted response
    """
    match = re.search(r"ASSISTANT:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_after_second_assistant(text):
    """
    Extract text after the second occurrence of 'assistant'.
    
    Args:
        text: Input text
        
    Returns:
        str: Extracted text
    """
    # Find all occurrences of "assistant"
    split_text = text.split('assistant')
    if len(split_text) > 2:
        # Join everything after the second "assistant"
        return 'assistant'.join(split_text[2:]).strip()
    else:
        return text  # Return original text if no split found

def load_image(image_path):
    """
    Load an image from a path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_for_vl(image, target_size=(336, 336)):
    """
    Preprocess an image for vision-language models.
    
    Args:
        image: PIL Image or path to image
        target_size: Tuple of (height, width)
        
    Returns:
        PIL.Image: Processed image
    """
    if isinstance(image, str):
        image = load_image(image)
    
    if image is None:
        return None
    
    # Resize the image
    image = image.resize(target_size, Image.LANCZOS)
    
    return image

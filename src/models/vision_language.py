import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig
)

def load_vision_language_model(device=None):
    """
    Load the Qwen2.5-VL model and processor.
    
    Args:
        device: torch device
        
    Returns:
        tuple: (model, processor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model ID
    model_id = "Aranya31/DermQwen-7b-adapter"
    processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # BitsAndBytesConfig for int-4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization and optimized memory usage
    print(f"Loading vision-language model from {model_id}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        quantization_config=bnb_config
    ).to(device)
    
    # Load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(processor_id)
    
    return model, processor

def generate_response(model, processor, image, prompt_text, max_tokens=512):
    """
    Generate a response using the vision-language model.
    
    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        image: PIL Image
        prompt_text: Text prompt
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Generated response
    """
    device = next(model.parameters()).device
    
    # Prepare inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Generate output
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    # Decode and return
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)

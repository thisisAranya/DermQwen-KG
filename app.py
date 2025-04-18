import gradio as gr
import torch
from PIL import Image
import re
import os

# Import project modules
from src.models.dino_classifier import DinoVisionTransformerClassifier, predict_skin_disease
from src.models.vision_language import load_vision_language_model
from src.utils.kg_retriever import load_knowledge_graph, setup_retrieval_model
from src.utils.preprocessing import disease_labels, extract_after_second_assistant

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
print("Loading DINO classifier...")
dino_model = DinoVisionTransformerClassifier().to(device)
model_path = os.path.join("models", "best_model.pth")
if os.path.exists(model_path):
    dino_model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: Model file not found at {model_path}")
dino_model.eval()

print("Loading Qwen VL model...")
vision_model, processor = load_vision_language_model(device)

print("Loading knowledge graph...")
knowledge_graph = load_knowledge_graph("knowledge_graph.graphml")
retrieval_model, entity_embeddings = setup_retrieval_model(knowledge_graph)

def rag_query(query, image_path, softmax_threshold=0.9):
    """
    Run the RAG pipeline on an image with a query.
    """
    # Encode the query using the retrieval model
    query_embedding = retrieval_model.encode(query)
    
    # Find the best matching entity from the KG
    best_match = max(entity_embeddings.items(), 
                     key=lambda item: torch.nn.functional.cosine_similarity(
                         torch.tensor(query_embedding).unsqueeze(0), 
                         torch.tensor(item[1]).unsqueeze(0)
                     ).item())
    best_entity = best_match[0]
    
    # Retrieve relations associated with the best entity
    related_entities = [(target, data['relation']) 
                         for target, data in knowledge_graph[best_entity].items()]
    relation_text = " ".join([f"{best_entity} -({relation})-> {target}" 
                              for target, relation in related_entities])
    
    # Load image
    raw_image = Image.open(image_path)
    
    # Step 1: Get disease name from VL model
    prompt_text = "What is the name of the disease?"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    output = vision_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    disease_response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Match disease name from VL model response
    vl_disease = None
    for disease in disease_labels:
        if disease.lower() in disease_response.lower():
            vl_disease = disease
            break
    
    # Step 2: Get prediction from DINO classifier
    probability, predicted_disease = predict_skin_disease(dino_model, image_path)
    
    # Step 3: Choose source based on confidence
    if probability > softmax_threshold:
        disease_name = predicted_disease
        source = "Auxiliary Classifier"
    else:
        disease_name = vl_disease if vl_disease else "Unknown"
        source = "Vision Language Model"
    
    # Step 4: Generate detailed response
    prompt_text = (
        f"Using knowledge about {best_entity} and its relations ({relation_text}), "
        f"answer the question in detail: The name of the disease is {disease_name}. {query}"
    )
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    output = vision_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    detailed_response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    
    return source, disease_name, query, extract_after_second_assistant(detailed_response)

# Gradio interface
def run_rag_pipeline(image, query, softmax_threshold):
    return rag_query(query, image, softmax_threshold)

# Create and launch Gradio app
demo = gr.Interface(
    fn=run_rag_pipeline,
    inputs=[
        gr.Image(type="filepath", label="Upload Skin Image"),
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Query"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Softmax Threshold"),
    ],
    outputs=[
        gr.Textbox(label="Label Source"),
        gr.Textbox(label="Predicted Disease Name"),
        gr.Textbox(label="User Query"),
        gr.Textbox(label="Cleaned Assistant Response")
    ],
    title = "Skin Disease Diagnosis with Qwen, DinoV2, and KG-RAG",
    description="Upload an image and enter a question. The model will predict the disease and answer your query.",
)

if __name__ == "__main__":
    demo.launch(share=True)

# DermQwen-KG: Skin Disease Diagnosis System

A multimodal system for skin disease diagnosis using Retrieval-Augmented Generation (RAG) with Qwen2.5-VL and knowledge graphs.

## Overview

This project combines a vision-language model (Qwen2.5-VL) with a DINO-based classifier and a knowledge graph to diagnose skin diseases from images and answer medical queries. The system uses:

- **Qwen2.5-VL-7B**: Vision-language model for interpreting skin disease images
- **DINO Transformer**: Pre-trained vision transformer fine-tuned on DermNet dataset
- **Knowledge Graph**: Structured information about skin diseases and treatments
- **RAG Pipeline**: Matches queries to relevant knowledge graph information

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DermQwen-KG.git
cd DermQwen-KG

# Install dependencies
pip install -r requirements.txt
```

## Data

The system uses two datasets:
- A knowledge graph dataset (zxzzzzzzzzzzzzzz from Kaggle)
- A selective DermNet dataset for skin disease images

Instructions for downloading these datasets are in the `data/README.md` file.

## Usage

Run the Gradio app:

```bash
python app.py
```

This will launch a web interface where you can:
1. Upload a skin disease image
2. Enter a question about the disease
3. Adjust the confidence threshold for the classifier
4. Get predictions and answers based on the knowledge graph

## Model Pipeline

The system works as follows:
1. Image is processed by both Qwen2.5-VL and DINO classifier
2. Based on confidence threshold, the disease label is selected
3. The query is matched to relevant nodes in the knowledge graph
4. The system generates a response using the image, disease label, and knowledge graph

## License

[MIT License](LICENSE)

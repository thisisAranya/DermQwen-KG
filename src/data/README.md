# Dataset and Model Download Instructions

This project requires two Kaggle datasets and a pre-trained DINO model:

- **Knowledge Graph Dataset**: `chapkhabo/zxzzzzzzzzzzzzzz`
- **Selective DermNet Dataset**: `aranyasaha/selective-dermnet-for-llm`
- **DINO Model**: `aranyasaha/dino-model-trained-on-dermnet`

## Prerequisites

- **Kaggle Account**: Ensure you have a Kaggle account.
- **Python Environment**: Activate your virtual environment:
  ```powershell
  cd D:\ACI Files\customAPI_LLM
  .\venv\Scripts\activate
  ```
- **Kaggle API**: Install the Kaggle package:
  ```powershell
  pip install kaggle
  ```

## Download Instructions

### Option 1: Using Kaggle API (Recommended)

1. **Set Up Kaggle API Token**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account).
   - Under "API," click **Create New API Token**.
   - Open the downloaded `kaggle.json` and note your `username` and `key`.

2. **Run Download Script**:
   - Create `download_kaggle.py`:
     ```powershell
     notepad download_kaggle.py
     ```
   - Paste the following code, replacing `your_kaggle_username` and `your_kaggle_api_key` with your Kaggle credentials:
     ```python
     import os
     import json
     from kaggle.api.kaggle_api_extended import KaggleApi

     # Kaggle credentials
     kaggle_user_name = 'your_kaggle_username'  # Replace with your Kaggle username
     kaggle_user_key = 'your_kaggle_api_key'   # Replace with your Kaggle API key

     # Save Kaggle API token
     kaggle_dir = os.path.expanduser('~/.kaggle')
     os.makedirs(kaggle_dir, exist_ok=True)
     with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
         json.dump({"username": kaggle_user_name, "key": kaggle_user_key}, f)
     os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

     # Authenticate
     api = KaggleApi()
     api.authenticate()

     # Create directories
     os.makedirs('dataset', exist_ok=True)
     os.makedirs('models', exist_ok=True)

     # Download datasets
     api.dataset_download_files('chapkhabo/zxzzzzzzzzzzzzzz', path='.', unzip=True)
     api.dataset_download_files('aranyasaha/selective-dermnet-for-llm', path='./dataset', unzip=True)

     # Download DINO model
     api.model_download_files('aranyasaha/dino-model-trained-on-dermnet', path='./models', unzip=True)

     # Move model file (if needed)
     model_source = './models/best_model.pth'
     model_destination = './src/models'
     if os.path.exists(model_source):
         os.makedirs(model_destination, exist_ok=True)
         os.rename(model_source, os.path.join(model_destination, 'best_model.pth'))

     print('Datasets and model downloaded successfully')
     ```
   - Save and run:
     ```powershell
     python download_kaggle.py
     ```

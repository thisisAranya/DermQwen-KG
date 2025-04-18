# Dataset Instructions

This project requires two datasets from Kaggle:

1. **Knowledge Graph Dataset**: "zxzzzzzzzzzzzzzz"
2. **Selective DermNet Dataset**: "selective-dermnet-for-llm"

## Download Instructions

### Option 1: Using Kaggle API (Recommended)

1. Set up your Kaggle API token:
   - Go to your Kaggle account settings
   - Create a new API token
   - Save the kaggle.json file to ~/.kaggle/

2. Run the download script:
   ```bash
   python -c "
   import os
   import kaggle
   
   # Create data directory if it doesn't exist
   os.makedirs('dataset', exist_ok=True)
   
   # Download the knowledge graph dataset
   kaggle.api.dataset_download_files('chapkhabo/zxzzzzzzzzzzzzzz', path='./', unzip=True)
   
   # Download the selective DermNet dataset
   kaggle.api.dataset_download_files('aranyasaha/selective-dermnet-for-llm', path='./dataset', unzip=True)
   
   print('Datasets downloaded successfully')
   "
   ```

### Option 2: Manual Download

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Search for and download:
   - "chapkhabo/zxzzzzzzzzzzzzzz"
   - "aranyasaha/selective-dermnet-for-llm"
3. Extract the downloaded files to the appropriate directories:
   - Knowledge graph files to the root project directory
   - DermNet dataset to the `dataset` directory

## DINO Model

You also need to download the pre-trained DINO model:

```bash
kaggle models download aranyasaha/dino-model-trained-on-dermnet
```

Extract the model file to a location referenced in your configuration.

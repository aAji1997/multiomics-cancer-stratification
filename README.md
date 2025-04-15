# Multiomics Cancer Stratification

A project for analyzing and stratifying cancer samples using multi-omics data integration.

## Features

- Data extraction from various omics formats
- Integration of clinical, genomic, and other data types
- Analysis tools for cancer sample stratification
- Joint autoencoder for multi-omics data integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multiomics-cancer-stratification.git
cd multiomics-cancer-stratification

# Install required packages
pip install -r requirements.txt
```

## Dataset

Download the preprocessed multi-omics dataset using the Kaggle CLI:

```bash
# Make sure you have your kaggle API key in your home directory in a hidden .kaggle folder
# Download and unzip the preprocessed dataset
kaggle datasets download progpug314/multiomics-preprocessed -p data/
```

## Getting Started

1. Make sure you have the required packages installed
2. Download the preprocessed data via Kaggle as described above
3. Run the joint autoencoder training script

## Training the Joint Autoencoder

```bash
# Navigate to the modelling directory
cd modelling/autoencoder

# Run training for colorectal cancer data
python train_joint_ae.py --cancer_type colorec --data_path ../../data/prepared_data_both.joblib --output_dir ../../results/autoencoder

# Run training for pancreatic cancer data
python train_joint_ae.py --cancer_type panc --data_path ../../data/prepared_data_both.joblib --output_dir ../../results/autoencoder
```

Additional training parameters:
- `--modalities`: Comma-separated list of omics modalities to use (default: rnaseq,methylation,scnv,miRNA)
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 150)
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)


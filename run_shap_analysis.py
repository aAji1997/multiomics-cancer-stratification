#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SHAP Analysis for Multi-omics Cancer Stratification Models

This script runs the SHAP analysis and generates visualizations.

Author: AI Assistant
Date: 2023
"""

import os
import argparse
import subprocess


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SHAP Analysis for Multi-omics Cancer Stratification Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--data_path', type=str, default='data/prepared_data_both.joblib',
                        help='Path to the prepared data joblib file')
    parser.add_argument('--raw_data_dir', type=str, default='data/colorec/omics_data',
                        help='Directory containing raw omics data files')

    # Model paths
    parser.add_argument('--integrated_model_path', type=str,
                        default='results/integrated_model_improved/colorec/best_model_checkpoint.pt',
                        help='Path to the trained integrated GCN model')
    parser.add_argument('--autoencoder_model_path', type=str,
                        default='results/autoencoder/joint_ae_model_colorec.pth',
                        help='Path to the trained autoencoder model')

    # Embedding paths
    parser.add_argument('--integrated_embeddings_path', type=str,
                        default='results/integrated_model_improved/colorec/integrated_embeddings_colorec.joblib',
                        help='Path to the integrated model embeddings')
    parser.add_argument('--autoencoder_embeddings_path', type=str,
                        default='results/autoencoder/joint_ae_embeddings_colorec.joblib',
                        help='Path to the autoencoder embeddings')

    # Config paths
    parser.add_argument('--integrated_config_path', type=str,
                        default='config/integrated_model_best_train_config.txt',
                        help='Path to the integrated model config file')
    parser.add_argument('--autoencoder_config_path', type=str,
                        default='config/autoencoder_model_best_train_config.txt',
                        help='Path to the autoencoder model config file')
    parser.add_argument('--modality_latents_path', type=str,
                        default='config/modality_latents_large.json',
                        help='Path to the modality latents config file')

    # Analysis parameters
    parser.add_argument('--cancer_type', type=str, default='colorec',
                        help='Cancer type to analyze')
    parser.add_argument('--num_background_samples', type=int, default=50,
                        help='Number of background samples for SHAP analysis')
    parser.add_argument('--num_test_samples', type=int, default=10,
                        help='Number of test samples for SHAP analysis')
    parser.add_argument('--modalities', type=str, default='rnaseq,methylation,scnv,miRNA',
                        help='Comma-separated list of modalities to analyze')
    parser.add_argument('--output_dir', type=str, default='results/shap_analysis',
                        help='Directory to save SHAP analysis results')
    parser.add_argument('--top_n_features', type=int, default=20,
                        help='Number of top features to display in summary plots')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')

    return parser.parse_args()


def main():
    """Main function to run SHAP analysis."""
    # Parse arguments
    args = parse_args()

    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "gene_level"), exist_ok=True)

    # Create analysis directory if it doesn't exist
    os.makedirs("analysis", exist_ok=True)

    # Run SHAP analysis
    print("Running SHAP analysis...")
    shap_cmd = [
        "python", "analysis/shap_analysis.py",
        "--data_path", args.data_path,
        "--raw_data_dir", args.raw_data_dir,
        "--integrated_model_path", args.integrated_model_path,
        "--autoencoder_model_path", args.autoencoder_model_path,
        "--integrated_embeddings_path", args.integrated_embeddings_path,
        "--autoencoder_embeddings_path", args.autoencoder_embeddings_path,
        "--integrated_config_path", args.integrated_config_path,
        "--autoencoder_config_path", args.autoencoder_config_path,
        "--modality_latents_path", args.modality_latents_path,
        "--cancer_type", args.cancer_type,
        "--num_background_samples", str(args.num_background_samples),
        "--num_test_samples", str(args.num_test_samples),
        "--modalities", args.modalities,
        "--output_dir", args.output_dir,
        "--top_n_features", str(args.top_n_features),
        "--device", args.device
    ]

    subprocess.run(shap_cmd)

    # Run gene mapper
    print("\nRunning gene mapper...")
    gene_mapper_cmd = [
        "python", "analysis/gene_mapper.py",
        "--shap_results", os.path.join(args.output_dir, "shap_results.joblib"),
        "--data_path", args.data_path,
        "--cancer_type", args.cancer_type,
        "--output_dir", os.path.join(args.output_dir, "gene_level"),
        "--top_n", str(args.top_n_features)
    ]

    subprocess.run(gene_mapper_cmd)

    print("\nSHAP analysis and gene mapping completed successfully!")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

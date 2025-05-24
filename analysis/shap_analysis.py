#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP Analysis for Multi-omics Cancer Stratification Models

This script performs SHAP (SHapley Additive exPlanations) analysis on the
integrated GCN model and autoencoder model to assess the contribution of
biomarkers to the final embeddings.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import json
import shap
import gc
import pandas as pd
import torch.optim as optim
import wandb
import time
import traceback
import shutil
import importlib.util
from datetime import datetime

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import project modules
from modelling.gcn.integrated_transformer_gcn_model import IntegratedTransformerGCN
from modelling.autoencoder.model import JointAutoencoder
from modelling.autoencoder.data_utils import load_prepared_data, prepare_graph_data
from modelling.gcn.train_integrated_transformer_gcn import (
    load_raw_omics_data, load_gene_embeddings, load_gene_interactions_and_masks,
    load_checkpoint
)
from modelling.gcn.train_integrated_transformer_gcn_improved import create_patient_gene_edges
from torch_geometric.utils import dense_to_sparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SHAP Analysis for Multi-omics Cancer Stratification Models',
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
    parser.add_argument('--use_saved_embeddings_only', action='store_true',
                        help='Skip model loading and use saved embeddings directly')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')

    return parser.parse_args()


def initialize_wandb(args):
    """
    Initialize Weights and Biases for experiment tracking.
    
    Args:
        args: Command line arguments
        
    Returns:
        wandb.Run: The initialized wandb run
    """
    # Load API key from config
    try:
        with open('api_config.json', 'r') as f:
            api_config = json.load(f)
        wandb_api_key = api_config.get('wandb_api_key')
        
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            print("Successfully logged in to Weights and Biases")
        else:
            print("Warning: No wandb_api_key found in api_config.json")
    except Exception as e:
        print(f"Warning: Could not load wandb API key from config: {e}")
    
    # Initialize wandb run
    run = wandb.init(
        project="multiomics-cancer-shap-analysis",
        name=f"shap_analysis_{args.cancer_type}_{wandb.util.generate_id()}",
        config={
            "cancer_type": args.cancer_type,
            "num_background_samples": args.num_background_samples,
            "num_test_samples": args.num_test_samples,
            "modalities": args.modalities,
            "top_n_features": args.top_n_features,
            "device": args.device,
            "use_saved_embeddings_only": args.use_saved_embeddings_only,
            "integrated_model_path": args.integrated_model_path,
            "autoencoder_model_path": args.autoencoder_model_path,
            "data_path": args.data_path
        },
        tags=["shap", "explainability", args.cancer_type]
    )
    
    return run


def upload_shap_results_to_wandb(args, shap_results, modality_indices):
    """
    Upload SHAP analysis results to Weights and Biases.
    
    Args:
        args: Command line arguments
        shap_results: Dictionary containing SHAP analysis results
        modality_indices: Dictionary mapping modality names to feature indices
    """
    print("Uploading SHAP analysis results to Weights and Biases...")
    
    # Create a summary table of top features for each model
    integrated_shap_values = shap_results['integrated_model']['shap_values']
    autoencoder_shap_values = shap_results['autoencoder_model']['shap_values']
    feature_names = shap_results['integrated_model']['feature_names']
    
    # Calculate mean absolute SHAP values for feature importance
    integrated_importance = np.mean(np.abs(integrated_shap_values), axis=0)
    autoencoder_importance = np.mean(np.abs(autoencoder_shap_values), axis=0)
    
    # Create feature importance tables
    feature_importance_data = []
    for i, feature_name in enumerate(feature_names):
        # Extract modality and gene index from feature name
        modality = feature_name.split('_')[0]
        gene_idx = feature_name.split('_')[1] if len(feature_name.split('_')) > 1 else str(i)
        
        feature_importance_data.append({
            "feature_name": feature_name,
            "modality": modality,
            "gene_index": gene_idx,
            "integrated_importance": float(integrated_importance[i]),
            "autoencoder_importance": float(autoencoder_importance[i]),
            "importance_difference": float(integrated_importance[i] - autoencoder_importance[i])
        })
    
    # Sort by integrated model importance
    feature_importance_data.sort(key=lambda x: x['integrated_importance'], reverse=True)
    
    # Create wandb table
    columns = ["feature_name", "modality", "gene_index", "integrated_importance", 
               "autoencoder_importance", "importance_difference"]
    feature_table = wandb.Table(columns=columns, data=[
        [row[col] for col in columns] for row in feature_importance_data[:50]  # Top 50 features
    ])
    
    # Log the feature importance table
    wandb.log({"feature_importance_top50": feature_table})
    
    # Calculate and log modality-level importance
    modality_importance = {}
    for modality, indices in modality_indices.items():
        integrated_mod_importance = np.mean([integrated_importance[i] for i in indices])
        autoencoder_mod_importance = np.mean([autoencoder_importance[i] for i in indices])
        
        modality_importance[modality] = {
            "integrated_importance": float(integrated_mod_importance),
            "autoencoder_importance": float(autoencoder_mod_importance),
            "num_features": len(indices)
        }
        
        # Log individual modality metrics
        wandb.log({
            f"modality_importance/{modality}/integrated": integrated_mod_importance,
            f"modality_importance/{modality}/autoencoder": autoencoder_mod_importance,
            f"modality_importance/{modality}/difference": integrated_mod_importance - autoencoder_mod_importance
        })
    
    # Log overall statistics
    wandb.log({
        "total_features": len(feature_names),
        "num_test_samples": len(shap_results['integrated_model']['test_data']),
        "num_background_samples": len(shap_results['integrated_model']['background_indices']),
        "top_integrated_feature_importance": float(np.max(integrated_importance)),
        "top_autoencoder_feature_importance": float(np.max(autoencoder_importance)),
        "mean_integrated_importance": float(np.mean(integrated_importance)),
        "mean_autoencoder_importance": float(np.mean(autoencoder_importance))
    })
    
    # Upload all generated plots as wandb artifacts
    plot_files = [
        "integrated_model_summary_plot.png",
        "autoencoder_model_summary_plot.png"
    ]
    
    # Add modality-specific plots
    for modality in modality_indices.keys():
        plot_files.extend([
            f"integrated_model_{modality}_plot.png",
            f"autoencoder_model_{modality}_plot.png"
        ])
    
    # Log plots as images
    for plot_file in plot_files:
        plot_path = os.path.join(args.output_dir, plot_file)
        if os.path.exists(plot_path):
            wandb.log({plot_file.replace('.png', ''): wandb.Image(plot_path)})
    
    # Create and upload SHAP results artifact
    artifact = wandb.Artifact(
        name=f"shap_results_{args.cancer_type}",
        type="analysis_results",
        description=f"SHAP analysis results for {args.cancer_type} cancer stratification models"
    )
    
    # Add the joblib results file
    results_path = os.path.join(args.output_dir, "shap_results.joblib")
    if os.path.exists(results_path):
        artifact.add_file(results_path)
    
    # Add all plot files
    for plot_file in plot_files:
        plot_path = os.path.join(args.output_dir, plot_file)
        if os.path.exists(plot_path):
            artifact.add_file(plot_path)
    
    # Upload the artifact
    wandb.log_artifact(artifact)
    
    # Create a summary report
    summary_html = f"""
    <h2>SHAP Analysis Summary for {args.cancer_type.title()} Cancer</h2>
    
    <h3>Dataset Statistics</h3>
    <ul>
        <li>Total Features: {len(feature_names)}</li>
        <li>Test Samples: {len(shap_results['integrated_model']['test_data'])}</li>
        <li>Background Samples: {len(shap_results['integrated_model']['background_indices'])}</li>
        <li>Modalities: {', '.join(modality_indices.keys())}</li>
    </ul>
    
    <h3>Top 10 Most Important Features (Integrated Model)</h3>
    <table border="1">
        <tr><th>Rank</th><th>Feature</th><th>Modality</th><th>Importance</th></tr>
    """
    
    for i, row in enumerate(feature_importance_data[:10]):
        summary_html += f"""
        <tr>
            <td>{i+1}</td>
            <td>{row['feature_name']}</td>
            <td>{row['modality']}</td>
            <td>{row['integrated_importance']:.4f}</td>
        </tr>
        """
    
    summary_html += """
    </table>
    
    <h3>Modality-Level Importance Comparison</h3>
    <table border="1">
        <tr><th>Modality</th><th>Integrated Model</th><th>Autoencoder Model</th><th>Difference</th></tr>
    """
    
    for modality, importance in modality_importance.items():
        difference = importance['integrated_importance'] - importance['autoencoder_importance']
        summary_html += f"""
        <tr>
            <td>{modality}</td>
            <td>{importance['integrated_importance']:.4f}</td>
            <td>{importance['autoencoder_importance']:.4f}</td>
            <td>{difference:.4f}</td>
        </tr>
        """
    
    summary_html += "</table>"
    
    # Log the HTML summary
    wandb.log({"analysis_summary": wandb.Html(summary_html)})
    
    print("Successfully uploaded SHAP analysis results to Weights and Biases!")


def load_config(config_path):
    """
    Load model configuration from a text file.

    Args:
        config_path: Path to the config file

    Returns:
        dict: Configuration parameters
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split(':', 1)
                # Try to convert to appropriate type
                try:
                    # Check if it's a boolean
                    if value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    # Check if it's a number
                    elif value.replace('.', '', 1).isdigit():
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    # Check if it's a string with quotes
                    elif value.startswith('"') and value.endswith('"'):
                        config[key] = value[1:-1]
                    else:
                        config[key] = value
                except ValueError:
                    config[key] = value
    return config


def inspect_checkpoint_architecture(checkpoint_path):
    """
    Inspect the checkpoint to understand the saved model architecture.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        dict: Architecture information extracted from the checkpoint
    """
    print(f"\nInspecting checkpoint architecture from {checkpoint_path}")
    
    # Load checkpoint on CPU to inspect
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_args = checkpoint.get('args', {})
    else:
        state_dict = checkpoint
        saved_args = {}
    
    print(f"Found {len(state_dict)} parameters in checkpoint")
    
    # Extract architecture information from the state dict
    arch_info = {
        'decoder_params': {},
        'modality_decoders': {},
        'transformer_params': {},
        'gcn_params': {},
        'has_dimension_adapters': False,
        'saved_args': saved_args,
        'exact_decoder_architecture': {}
    }
    
    # Analyze decoder architecture in detail
    for key, tensor in state_dict.items():
        if 'omics_decoder.patient_decoder.0.weight' in key:
            arch_info['decoder_params']['patient_decoder_input'] = tensor.shape[1]
            arch_info['decoder_params']['patient_decoder_hidden'] = tensor.shape[0]
            arch_info['exact_decoder_architecture']['patient_decoder_hidden'] = tensor.shape[0]
            print(f"Patient decoder: input={tensor.shape[1]}, hidden={tensor.shape[0]}")
            
        elif 'omics_decoder.patient_decoder.3.weight' in key:
            arch_info['exact_decoder_architecture']['patient_decoder_output'] = tensor.shape[0]
            print(f"Patient decoder output: {tensor.shape[0]}")
            
        elif 'omics_decoder.reconstruction_mlp.0.weight' in key:
            arch_info['decoder_params']['reconstruction_input'] = tensor.shape[1]
            arch_info['decoder_params']['reconstruction_output'] = tensor.shape[0]
            arch_info['exact_decoder_architecture']['reconstruction_hidden'] = tensor.shape[0]
            print(f"Reconstruction MLP: input={tensor.shape[1]}, output={tensor.shape[0]}")
            
        elif 'omics_decoder.reconstruction_mlp.3.weight' in key:
            arch_info['exact_decoder_architecture']['reconstruction_output'] = tensor.shape[0]
            print(f"Reconstruction final output: {tensor.shape[0]}")
            
        elif 'omics_decoder.modality_decoders.' in key and '.0.weight' in key:
            # Extract modality name
            modality = key.split('omics_decoder.modality_decoders.')[1].split('.0.weight')[0]
            arch_info['modality_decoders'][modality] = {
                'input_dim': tensor.shape[1],
                'hidden_dim': tensor.shape[0]
            }
            print(f"Modality decoder {modality}: input={tensor.shape[1]}, hidden={tensor.shape[0]}")
            
        elif '_dim_adapter' in key:
            arch_info['has_dimension_adapters'] = True
            modality = key.split('omics_decoder.')[1].split('_dim_adapter')[0]
            if 'weight' in key:
                print(f"Dimension adapter for {modality}: {tensor.shape}")
    
    # Extract modality latent dimensions from modality decoders
    modality_latents = {}
    modality_decoder_hidden_dims = {}
    for modality, decoder_info in arch_info['modality_decoders'].items():
        modality_latents[modality] = decoder_info['input_dim']
        modality_decoder_hidden_dims[modality] = decoder_info['hidden_dim']
    
    arch_info['inferred_modality_latents'] = modality_latents
    arch_info['exact_decoder_architecture']['modality_decoder_hidden_dims'] = modality_decoder_hidden_dims
    
    print(f"Inferred modality latent dimensions: {modality_latents}")
    print(f"Exact modality decoder hidden dimensions: {modality_decoder_hidden_dims}")
    
    return arch_info


def load_integrated_model(args, integrated_config, prepared_data):
    """
    Load the integrated transformer GCN model.
    This version inspects the checkpoint first to match the exact architecture.

    Args:
        args: Command line arguments
        integrated_config: Model configuration
        prepared_data: Prepared data dictionary

    Returns:
        tuple: (model, gene_embeddings, edge_index_dict, raw_omics_data_dict, patient_ids, gene_list)
    """
    device = torch.device(args.device)

    # --- First, inspect the checkpoint to understand the actual architecture --- #
    arch_info = inspect_checkpoint_architecture(args.integrated_model_path)
    
    # Use saved args if available, otherwise fall back to integrated_config
    saved_args = arch_info['saved_args']
    if saved_args:
        print("Using configuration from saved checkpoint")
        config = saved_args
    else:
        print("Using provided integrated_config (checkpoint has no saved args)")
        config = integrated_config
    
    # Override modality latent dimensions if we inferred them from the checkpoint
    if arch_info['inferred_modality_latents']:
        print(f"Using modality latent dimensions from checkpoint: {arch_info['inferred_modality_latents']}")
        modality_latent_dims = arch_info['inferred_modality_latents']
    else:
        # Try to load from config file
        modality_latent_dims = None
        if hasattr(args, 'modality_latents_path') and args.modality_latents_path:
            try:
                with open(args.modality_latents_path, 'r') as f:
                    modality_latent_dims = json.load(f)
                modality_latent_dims = {k.lower(): v for k, v in modality_latent_dims.items()}
                print(f"Using modality latent dimensions from config file: {modality_latent_dims}")
            except Exception as e:
                print(f"Error loading modality latents from config: {e}")
                modality_latent_dims = None

    # --- Load Data (following the improved training script approach) --- #
    # Load Raw Omics
    raw_omics_data_dict, patient_ids = load_raw_omics_data(args.data_path, args.cancer_type)
    if raw_omics_data_dict is None:
        print("Error: Failed to load raw omics data")
        return None

    # Ensure modality names are standardized (lowercase) for consistent matching with config
    raw_omics_data_dict_standardized = {k.lower(): v for k, v in raw_omics_data_dict.items()}
    raw_omics_data_dict = raw_omics_data_dict_standardized

    omics_input_dims = {k: v.shape[1] for k, v in raw_omics_data_dict.items()}
    print(f"Raw Omics Input Dims: {omics_input_dims}")
    print(f"Modalities found in the data (standardized to lowercase): {list(omics_input_dims.keys())}")

    # Move raw omics data to device early
    for k in raw_omics_data_dict:
        raw_omics_data_dict[k] = raw_omics_data_dict[k].to(device)
    print(f"Moved raw omics data to {device}")

    # Load Gene Embeddings
    gene_embeddings, gene_emb_list = load_gene_embeddings(
        config.get('gene_embedding_path', args.autoencoder_embeddings_path)
    )
    if gene_embeddings is None:
        print("Error: Failed to load gene embeddings")
        return None
    gene_embeddings = gene_embeddings.to(device)
    print(f"Moved gene embeddings to {device}")

    # Store original gene embeddings for structure preservation loss
    original_gene_embeddings = gene_embeddings.clone().detach()

    # Allow gene embeddings to be updated with controlled learning
    if config.get('unfreeze_gene_embeddings', False):
        gene_embeddings.requires_grad_(True)
        print("Using controlled updates to gene embeddings with structure preservation.")
    else:
        # Keep frozen as in original implementation
        gene_embeddings.requires_grad_(False)
        print("Frozen pre-trained gene embeddings. They will not be updated during GCN training.")

    # Load Gene Interactions and Masks
    adj_matrix_gg, interaction_gene_list, gene_masks = load_gene_interactions_and_masks(
        args.data_path, args.cancer_type
    )
    if adj_matrix_gg is None:
        print("Error: Failed to load gene interactions")
        return None

    # --- Gene List Consistency Check (exact same logic as improved training script) --- #
    if gene_emb_list != interaction_gene_list:
        print("\nCritical Warning: Gene lists from gene embeddings and gene interactions do NOT match!")
        # Attempt to align if possible, otherwise error or proceed with caution
        common_genes = sorted(list(set(gene_emb_list) & set(interaction_gene_list)))
        if not common_genes:
             print("Error: No common genes between embedding list and interaction list. Cannot proceed.")
             return None
        print(f"Found {len(common_genes)} common genes. Re-indexing embeddings and adjacency matrix...")

        # Re-index Embeddings
        gene_map_emb = {gene: i for i, gene in enumerate(gene_emb_list)}
        common_indices_emb = [gene_map_emb[g] for g in common_genes]
        gene_embeddings = gene_embeddings[common_indices_emb, :]

        # Re-index Adjacency Matrix (more complex for sparse/dense)
        gene_map_interact = {gene: i for i, gene in enumerate(interaction_gene_list)}
        common_indices_interact = [gene_map_interact[g] for g in common_genes]

        if hasattr(adj_matrix_gg, "shape") and adj_matrix_gg.shape[0] == adj_matrix_gg.shape[1]:
            if hasattr(adj_matrix_gg, "tocsr"): # Sparse
                 adj_matrix_gg = adj_matrix_gg.tocsr()[common_indices_interact, :][:, common_indices_interact]
            else: # Assume dense numpy/torch tensor
                 adj_matrix_gg = adj_matrix_gg[np.ix_(common_indices_interact, common_indices_interact)]
        else:
             print("Warning: Could not safely re-index adjacency matrix. Shape mismatch or unknown format.")

        gene_list = common_genes # Use the common gene list going forward
        print(f"Aligned data to {len(gene_list)} common genes.")
        print(f"New gene embeddings shape: {gene_embeddings.shape}")
        if hasattr(adj_matrix_gg, "shape"): print(f"New adj matrix shape: {adj_matrix_gg.shape}")

        # Re-index gene masks if they exist
        if gene_masks is not None:
            print("Re-indexing gene masks...")
            aligned_masks = {}
            for modality, mask_list in gene_masks.items():
                # Assuming mask_list is aligned with interaction_gene_list
                if len(mask_list) == len(interaction_gene_list):
                    original_mask = np.array(mask_list)
                    aligned_mask_modality = original_mask[common_indices_interact]
                    aligned_masks[modality] = aligned_mask_modality.tolist()
                else:
                    print(f"Warning: Dimension mismatch for mask '{modality}'. Skipping mask alignment.")
            gene_masks = aligned_masks
            print(f"Aligned masks for modalities: {list(gene_masks.keys())}")
    else:
        gene_list = gene_emb_list # Lists match, use either one
        print("\nGene lists from embeddings and interactions match.")

    # Validate gene masks dimension after potential alignment
    if gene_masks:
        for modality, mask in gene_masks.items():
             if len(mask) != len(gene_list):
                 print(f"Error: Post-alignment dimension mismatch for mask '{modality}' ({len(mask)}) vs gene list ({len(gene_list)}).")
                 print("Masks will be ignored.")
                 gene_masks = None
                 break

    # Number of genes is needed for model initialization
    num_genes = len(gene_list)

    # --- Construct Graph Edge Index Dictionary (exact same logic as improved training script) --- #
    print("\nConstructing graph edges...")
    edge_index_dict = {}

    # Gene-Gene Edges
    if adj_matrix_gg is not None:
        try:
            if hasattr(adj_matrix_gg, "tocoo"):
                coo = adj_matrix_gg.tocoo()
                edge_index_gg = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
            else: # Assume numpy/torch tensor
                adj_tensor_gg = torch.tensor(adj_matrix_gg, dtype=torch.float32)
                edge_index_gg, _ = dense_to_sparse(adj_tensor_gg)
            edge_index_dict[('gene', 'interacts', 'gene')] = edge_index_gg.to(device)
            print(f"Added {edge_index_gg.shape[1]} gene-gene edges.")
        except Exception as e:
            print(f"Error processing gene adjacency matrix: {e}. Skipping gene-gene edges.")
            edge_index_dict[('gene', 'interacts', 'gene')] = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
         edge_index_dict[('gene', 'interacts', 'gene')] = torch.empty((2, 0), dtype=torch.long, device=device)

    # Patient-Gene Edges (Requires raw omics data for linking)
    omics_df_for_links = None
    pg_link_omics = config.get('pg_link_omics', None)
    if pg_link_omics and pg_link_omics in raw_omics_data_dict:
         print(f"Using raw {pg_link_omics} tensor for patient-gene links...")
         omics_tensor_pg = raw_omics_data_dict[pg_link_omics].cpu() # Move to CPU for pandas
         # Create DataFrame with correct index/columns
         omics_df_for_links = pd.DataFrame(omics_tensor_pg.numpy(), index=patient_ids, columns=gene_list[:omics_tensor_pg.shape[1]])

         # Check if gene list used for columns matches the full gene_list
         if omics_df_for_links.shape[1] != len(gene_list):
              print(f"Warning: Number of features in {pg_link_omics} ({omics_df_for_links.shape[1]}) does not match final gene list ({len(gene_list)}). Cannot create P-G links accurately this way.")
              omics_df_for_links = None

    if omics_df_for_links is not None:
        edge_index_pg = create_patient_gene_edges(
            omics_df_for_links, patient_ids, gene_list,
            link_type=config.get('pg_link_type', 'threshold'),
            threshold=config.get('pg_link_threshold', 0.5),
            top_k=config.get('pg_link_top_k', 10)
        )
        edge_index_dict[('patient', 'expresses', 'gene')] = edge_index_pg.to(device)
        if edge_index_pg.numel() > 0:
            edge_index_gp = edge_index_pg[[1, 0], :]
            edge_index_dict[('gene', 'rev_expresses', 'patient')] = edge_index_gp.to(device)
            print(f"Added {edge_index_pg.shape[1]} patient-gene edges and {edge_index_gp.shape[1]} reverse edges.")
        else:
            edge_index_dict[('gene', 'rev_expresses', 'patient')] = torch.empty((2, 0), dtype=torch.long, device=device)
            print("No patient-gene edges created.")
    else:
        print("Skipping patient-gene edge creation (required omics not available or mismatched).")
        edge_index_dict[('patient', 'expresses', 'gene')] = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_index_dict[('gene', 'rev_expresses', 'patient')] = torch.empty((2, 0), dtype=torch.long, device=device)

    # Define Metadata
    node_types = ['patient', 'gene']
    metadata = (node_types, list(edge_index_dict.keys()))
    print("\nGraph Metadata:", metadata)

    # --- Model Instantiation (with architecture matching the checkpoint) --- #
    print("\nInstantiating IntegratedTransformerGCN model to match checkpoint architecture...")

    # Use custom model creation that exactly matches the checkpoint
    model = create_model_matching_checkpoint(
        arch_info, config, omics_input_dims, metadata, gene_embeddings, num_genes, device
    )

    # --- Initialize Lazy Modules (if any) --- #
    print("\nInitializing lazy modules with a single batch...")
    
    # Create a small batch to initialize dynamic modules
    batch_size = min(4, len(patient_ids))
    batch_indices = list(range(batch_size))
    
    # Extract batch data for ALL modalities
    batch_omics_data = {}
    for modality, tensor in raw_omics_data_dict.items():
        batch_omics_data[modality] = tensor[batch_indices]
    
    # Initialize lazy modules with a dummy forward pass
    with torch.no_grad():
        try:
            _ = model(batch_omics_data, gene_embeddings, edge_index_dict)
            print("Lazy modules initialized successfully with single batch.")
        except Exception as e:
            print(f"Warning: Error initializing lazy modules: {e}")
            # Force initialization of all LazyModules
            for _, module in model.named_modules():
                if hasattr(module, 'reset_parameters'):
                    try:
                        module.reset_parameters()
                    except:
                        pass

    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Create a dummy optimizer (required for load_checkpoint)
    if config.get('unfreeze_gene_embeddings', False):
        optimizer = optim.Adam([
            {'params': [p for _, p in model.named_parameters()], 'lr': config.get('gcn_lr', 0.001)},
            {'params': [gene_embeddings], 'lr': config.get('gene_lr', 0.0001)}
        ], weight_decay=config.get('gcn_weight_decay', 0.0001))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.get('gcn_lr', 0.001), 
                             weight_decay=config.get('gcn_weight_decay', 0.0001))

    # Load checkpoint using the proper function with remapping (this will not print "Loading checkpoint..." again)
    print(f"\nLoading model weights from checkpoint...")
    try:
        start_epoch, training_losses, loaded_gene_embeddings, _ = load_checkpoint_with_remapping(
            args.integrated_model_path, model, optimizer, device, scaler=None
        )

        # Restore gene embeddings if they were trainable and saved
        if config.get('unfreeze_gene_embeddings', False) and loaded_gene_embeddings is not None:
            print("Restoring trainable gene embeddings from checkpoint")
            gene_embeddings = loaded_gene_embeddings.to(device)
            gene_embeddings.requires_grad_(True)
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # Set model to evaluation mode
    model.eval()

    print("Integrated model loaded successfully!")

    return model, gene_embeddings, edge_index_dict, raw_omics_data_dict, patient_ids, gene_list


def load_autoencoder_model(args, autoencoder_config, prepared_data):
    """
    Load the joint autoencoder model.

    Args:
        args: Command line arguments
        autoencoder_config: Model configuration
        prepared_data: Prepared data dictionary

    Returns:
        tuple: (model, graph_node_features, graph_edge_index, graph_edge_weight, omics_data_dict, patient_ids, gene_list)
    """
    device = torch.device(args.device)

    # Extract data for the specified cancer type
    cancer_data = prepared_data[args.cancer_type]
    omics_data_dict = cancer_data['omics_data']
    adj_matrix = cancer_data['adj_matrix']
    gene_list = cancer_data['gene_list']
    gene_masks = cancer_data.get('gene_masks', None)

    # Get patient IDs
    patient_ids = []
    for modality in omics_data_dict.values():
        if 'patient_id' in modality.columns:
            patient_ids = modality['patient_id'].tolist()
            break

    # Prepare graph data
    graph_node_features, graph_edge_index, graph_edge_weight, graph_adj_tensor = prepare_graph_data(
        adj_matrix, node_init_modality='identity'
    )

    # Move graph data to device
    graph_node_features = graph_node_features.to(device)
    graph_edge_index = graph_edge_index.to(device)
    graph_edge_weight = graph_edge_weight.to(device)
    graph_adj_tensor = graph_adj_tensor.to(device)

    # Load modality latents
    with open(args.modality_latents_path, 'r') as f:
        modality_latent_dims = json.load(f)

    # Parse modalities and standardize names
    modalities_raw = args.modalities.split(',')
    # Standardize modality names to match the saved model
    modalities_to_use = []
    for modality in modalities_raw:
        if modality.lower() == 'mirna':
            # Use 'mirna' (lowercase) to match the saved model
            modalities_to_use.append('mirna')
        else:
            modalities_to_use.append(modality)
    print(f"Using standardized modality names: {modalities_to_use}")

    # Initialize model
    model = JointAutoencoder(
        num_nodes=len(gene_list),
        modality_latent_dims=modality_latent_dims,
        modality_order=modalities_to_use,
        graph_feature_dim=graph_node_features.shape[1],
        gene_embedding_dim=autoencoder_config.get('gene_embedding_dim', 64),
        patient_embedding_dim=autoencoder_config.get('patient_embedding_dim', 128),
        graph_dropout=autoencoder_config.get('graph_dropout', 0.5),
        gene_masks=gene_masks
    ).to(device)

    # Create a custom wrapper class for SHAP analysis
    class ShapCompatibleAutoencoder(nn.Module):
        """
        A wrapper class that mimics the JointAutoencoder but is compatible with the saved model weights.
        This class only implements the methods needed for SHAP analysis.
        """
        def __init__(self, base_model, state_dict):
            super().__init__()
            self.base_model = base_model
            # Copy all attributes from the base model
            for attr_name in dir(base_model):
                if not attr_name.startswith('__') and not callable(getattr(base_model, attr_name)):
                    setattr(self, attr_name, getattr(base_model, attr_name))

            # Load the state dict directly into this wrapper
            self.load_state_dict(state_dict, strict=False)

        def forward(self, *args, **kwargs):
            return self.base_model.forward(*args, **kwargs)

        def encode(self, *args, **kwargs):
            return self.base_model.encode(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.base_model.decode(*args, **kwargs)

        def graph_autoencoder(self, *args, **kwargs):
            return self.base_model.graph_autoencoder(*args, **kwargs)

        def omics_processor(self, *args, **kwargs):
            return self.base_model.omics_processor(*args, **kwargs)

    # Load model weights
    try:
        # Load the checkpoint file
        checkpoint = torch.load(args.autoencoder_model_path, map_location=device)

        # Get the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is a checkpoint dictionary, extract the model_state_dict
            print(f"Loading autoencoder from checkpoint format file")
            state_dict = checkpoint['model_state_dict']
        else:
            # Try loading directly (older format)
            state_dict = checkpoint

        # Handle modality name differences (miRNA vs mirna)
        # Create a new state dict with standardized keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Standardize modality names to lowercase
            new_key = k.replace('miRNA', 'mirna').replace('rnaseq', 'rnaseq').replace('scnv', 'scnv')
            new_state_dict[new_key] = v

        # First try loading the state dict directly to see what's incompatible
        incompatible_keys = model.load_state_dict(new_state_dict, strict=False)

        # Print information about incompatible keys
        if incompatible_keys.missing_keys:
            print(f"Warning: {len(incompatible_keys.missing_keys)} missing keys in autoencoder state dict")
            print(f"First few missing keys: {incompatible_keys.missing_keys[:3]}")

        if incompatible_keys.unexpected_keys:
            print(f"Warning: {len(incompatible_keys.unexpected_keys)} unexpected keys in autoencoder state dict")
            print(f"First few unexpected keys: {incompatible_keys.unexpected_keys[:3]}")

        # If there are size mismatches, create a wrapper model
        if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
            print("Creating a SHAP-compatible wrapper model for autoencoder to handle architecture differences...")
            model = ShapCompatibleAutoencoder(model, new_state_dict)
            print("Autoencoder wrapper model created successfully")

        print(f"Loaded autoencoder model weights from {args.autoencoder_model_path}")
    except Exception as e:
        print(f"Error loading autoencoder model weights: {e}")
        return None

    # Set model to evaluation mode
    model.eval()

    return model, graph_node_features, graph_edge_index, graph_edge_weight, omics_data_dict, patient_ids, gene_list


def create_model_matching_checkpoint(arch_info, config, omics_input_dims, metadata, gene_embeddings, num_genes, device):
    """
    Create a model that exactly matches the checkpoint architecture.
    
    Args:
        arch_info: Architecture information from checkpoint inspection
        config: Configuration dictionary
        omics_input_dims: Omics input dimensions
        metadata: Graph metadata
        gene_embeddings: Gene embeddings tensor
        num_genes: Number of genes
        device: Device to create model on
        
    Returns:
        IntegratedTransformerGCN model that matches the checkpoint
    """
    print("Creating model with exact checkpoint architecture matching...")
    
    # Extract exact dimensions from checkpoint
    exact_arch = arch_info['exact_decoder_architecture']
    patient_decoder_hidden = exact_arch.get('patient_decoder_hidden', 64)
    modality_decoder_hidden_dims = exact_arch.get('modality_decoder_hidden_dims', {})
    modality_latents = arch_info['inferred_modality_latents']
    
    print(f"Target patient decoder hidden: {patient_decoder_hidden}")
    print(f"Target modality decoder hidden dims: {modality_decoder_hidden_dims}")
    
    # Create model with minimal decoder first
    model = IntegratedTransformerGCN(
        # Transformer Args
        omics_input_dims=omics_input_dims,
        transformer_embed_dim=config.get('transformer_embed_dim', 128),
        transformer_num_heads=config.get('transformer_num_heads', 4),
        transformer_ff_dim=config.get('transformer_ff_dim', 256),
        num_transformer_layers=config.get('transformer_layers', 2),
        transformer_output_dim=config.get('transformer_output_dim', 64),
        transformer_dropout=config.get('transformer_dropout', 0.1),
        # GCN Args
        gcn_metadata=metadata,
        gene_feature_dim=gene_embeddings.shape[1],
        gcn_hidden_channels=config.get('gcn_hidden_dim', 256),
        gcn_out_channels=config.get('gcn_output_dim', 128),
        gcn_num_layers=config.get('gcn_layers', 2),
        gcn_conv_type=config.get('gcn_conv_type', 'sage'),
        gcn_num_heads=config.get('gcn_gat_heads', 4),
        gcn_dropout_rate=config.get('gcn_dropout', 0.5),
        gcn_use_layer_norm=not config.get('gcn_no_norm', False),
        gene_masks=None,  # Will add gene masks after loading
        add_omics_decoder=True,
        use_modality_specific_decoders=True,  # Force this to match checkpoint
        decoder_activation=config.get('decoder_activation', 'sigmoid'),
        decoder_patient_batch_size=16,
        decoder_mlp_factor=0.1,  # Use small factor initially to create minimal decoder
        genes_per_chunk=config.get('genes_per_chunk', 10),
        modality_latent_dims=modality_latents,  # Use exact latent dims from checkpoint
        num_genes=num_genes,
        use_gradient_checkpointing=False,
        extreme_memory_efficient=False,
        modality_by_modality=config.get('modality_by_modality', False)
    ).to(device)
    
    # Now replace the decoder with one that has exact dimensions
    print("Replacing decoder components with exact checkpoint dimensions...")
    
    gcn_patient_out_dim = config.get('gcn_output_dim', 128)
    gcn_gene_out_dim = config.get('gcn_output_dim', 128)
    
    # Replace patient decoder with exact dimensions
    model.omics_decoder.patient_decoder = nn.Sequential(
        nn.Linear(gcn_patient_out_dim, patient_decoder_hidden),
        nn.ReLU(),
        nn.LayerNorm(patient_decoder_hidden),
        nn.Linear(patient_decoder_hidden, patient_decoder_hidden)
    ).to(device)
    
    print(f"Created patient decoder: {gcn_patient_out_dim} -> {patient_decoder_hidden} -> {patient_decoder_hidden}")
    
    # Replace modality-specific decoders with exact dimensions
    model.omics_decoder.modality_decoders = nn.ModuleDict()
    for modality, latent_dim in modality_latents.items():
        hidden_dim = modality_decoder_hidden_dims.get(modality, 32)  # Default to 32 if not found
        
        model.omics_decoder.modality_decoders[modality] = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        print(f"Created modality decoder {modality}: {latent_dim} -> {hidden_dim} -> 1")
    
    # Replace reconstruction MLP with exact dimensions
    reconstruction_input = patient_decoder_hidden + gcn_gene_out_dim  # From checkpoint inspection
    reconstruction_hidden = exact_arch.get('reconstruction_hidden', 48)
    reconstruction_output = exact_arch.get('reconstruction_output', 64)  # Fixed: should be 64, not sum of latents
    
    model.omics_decoder.reconstruction_mlp = nn.Sequential(
        nn.Linear(reconstruction_input, reconstruction_hidden),
        nn.ReLU(),
        nn.LayerNorm(reconstruction_hidden),
        nn.Linear(reconstruction_hidden, reconstruction_output)
    ).to(device)
    
    print(f"Created reconstruction MLP: {reconstruction_input} -> {reconstruction_hidden} -> {reconstruction_output}")
    
    # Add dimension adapter if it exists in checkpoint
    if arch_info['has_dimension_adapters']:
        print("Adding dimension adapters found in checkpoint...")
        # Add methylation dimension adapter as found in checkpoint
        if 'methylation' in modality_latents:
            adapter_dim = modality_latents['methylation']  # 128 from checkpoint
            model.omics_decoder.methylation_dim_adapter = nn.Linear(adapter_dim, adapter_dim).to(device)
            print(f"Added methylation dimension adapter: {adapter_dim} -> {adapter_dim}")
    
    # Update the total latent dimension
    model.omics_decoder.total_latent_dim = sum(modality_latents.values())
    model.omics_decoder.modality_latent_dims = modality_latents
    
    print(f"Final model decoder architecture summary:")
    print(f"  Patient decoder: {gcn_patient_out_dim} -> {patient_decoder_hidden}")
    print(f"  Reconstruction MLP: {reconstruction_input} -> {reconstruction_hidden} -> {reconstruction_output}")
    print(f"  Modality decoders: {modality_decoder_hidden_dims}")
    print(f"  Total latent dimension: {model.omics_decoder.total_latent_dim}")
    
    return model


def remap_checkpoint_parameters(checkpoint_state_dict, model_state_dict):
    """
    Remap checkpoint parameters to match the current model's expected parameter names.
    This handles changes in PyTorch Geometric's parameter naming conventions.
    
    Args:
        checkpoint_state_dict: State dict from the saved checkpoint
        model_state_dict: State dict from the current model
        
    Returns:
        dict: Remapped state dict that matches the current model
    """
    print("Remapping checkpoint parameters to match current model...")
    
    # Create a new state dict for remapped parameters
    remapped_state_dict = {}
    
    # Get the current model's expected parameter names
    expected_keys = set(model_state_dict.keys())
    checkpoint_keys = set(checkpoint_state_dict.keys())
    
    print(f"Expected parameters: {len(expected_keys)}")
    print(f"Checkpoint parameters: {len(checkpoint_keys)}")
    
    # Create mapping for GCN parameter name changes
    # Old format: ('gene', 'interacts', 'gene') -> New format: <gene___interacts___gene>
    def convert_edge_type_name(old_key):
        """Convert old edge type format to new format"""
        if ".convs.(" in old_key and ")." in old_key:
            # Extract the edge type part
            before_edge = old_key.split(".convs.(")[0]
            edge_and_after = old_key.split(".convs.(")[1]
            edge_type_str = edge_and_after.split(").")[0]
            after_edge = edge_and_after.split(").")[1]
            
            # Parse the edge type tuple
            try:
                # Remove quotes and spaces, split by comma
                edge_type_clean = edge_type_str.replace("'", "").replace('"', '').replace(" ", "")
                src, rel, dst = edge_type_clean.split(",")
                # Create new format
                new_edge_type = f"<{src}___{rel}___{dst}>"
                new_key = f"{before_edge}.convs.{new_edge_type}.{after_edge}"
                return new_key
            except:
                return old_key
        return old_key
    
    # Process each parameter in the checkpoint
    for old_key, tensor in checkpoint_state_dict.items():
        # Try direct mapping first
        if old_key in expected_keys:
            remapped_state_dict[old_key] = tensor
            continue
            
        # Try converting GCN parameter names
        new_key = convert_edge_type_name(old_key)
        if new_key in expected_keys:
            remapped_state_dict[new_key] = tensor
            print(f"Remapped: {old_key} -> {new_key}")
            continue
            
        # If no mapping found, keep the original key (it might be handled elsewhere)
        remapped_state_dict[old_key] = tensor
    
    # Check for any remaining mismatches
    missing_keys = expected_keys - set(remapped_state_dict.keys())
    unexpected_keys = set(remapped_state_dict.keys()) - expected_keys
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} parameters still missing after remapping")
        if len(missing_keys) <= 10:  # Show only first 10 to avoid spam
            for key in list(missing_keys)[:10]:
                print(f"  Missing: {key}")
    
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected parameters after remapping")
        if len(unexpected_keys) <= 10:  # Show only first 10 to avoid spam  
            for key in list(unexpected_keys)[:10]:
                print(f"  Unexpected: {key}")
    
    return remapped_state_dict


def load_checkpoint_with_remapping(checkpoint_path, model, optimizer, device, scaler=None):
    """
    Load checkpoint with parameter remapping to handle naming convention changes.

                Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the checkpoint into
        optimizer: Optimizer to load state into
        device: Device to load tensors on
        scaler: Optional GradScaler for mixed precision

                Returns:
        tuple: (start_epoch, training_losses, loaded_gene_embeddings, checkpoint_data)
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
        start_epoch = checkpoint.get('epoch', 0) + 1
        training_losses = checkpoint.get('training_losses', [])
        loaded_gene_embeddings = checkpoint.get('gene_embeddings', None)
        
        # Load optimizer state if available and requested
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load scaler state if available
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("Loaded scaler state from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load scaler state: {e}")
                
    else:
        # Old format - just the state dict
        checkpoint_state_dict = checkpoint
        start_epoch = 0
        training_losses = []
        loaded_gene_embeddings = None
    
    # Remap the parameters to match current model
    model_state_dict = model.state_dict()
    remapped_state_dict = remap_checkpoint_parameters(checkpoint_state_dict, model_state_dict)
    
    # Load the remapped state dict
    try:
        incompatible_keys = model.load_state_dict(remapped_state_dict, strict=False)
        
        if incompatible_keys.missing_keys:
            print(f"Warning: {len(incompatible_keys.missing_keys)} missing keys after remapping")
            # Only show first few to avoid spam
            for key in incompatible_keys.missing_keys[:5]:
                print(f"  Missing: {key}")
        
        if incompatible_keys.unexpected_keys:
            print(f"Warning: {len(incompatible_keys.unexpected_keys)} unexpected keys after remapping")
            # Only show first few to avoid spam  
            for key in incompatible_keys.unexpected_keys[:5]:
                print(f"  Unexpected: {key}")
                
        print("Checkpoint loaded successfully with remapping!")
        
    except Exception as e:
        print(f"Error loading checkpoint even after remapping: {e}")
        raise e
    
    return start_epoch, training_losses, loaded_gene_embeddings, checkpoint


# Model wrapper classes for SHAP analysis (moved outside main for proper detection)
class IntegratedModelWrapper(nn.Module):
    """Wrapper for the integrated transformer GCN model for SHAP analysis."""
    
    def __init__(self, model, gene_embeddings, edge_index_dict, raw_omics_data_dict):
        super().__init__()
        self.model = model
        self.gene_embeddings = gene_embeddings
        self.edge_index_dict = edge_index_dict
        self.device = next(model.parameters()).device
        # Store modality dimensions for splitting input data
        self.modality_dims = {k: v.shape[1] for k, v in raw_omics_data_dict.items()}
        self.modality_order = list(raw_omics_data_dict.keys())
        print(f"Initialized IntegratedModelWrapper with modalities: {self.modality_order}")
        print(f"Modality dimensions: {self.modality_dims}")

    def forward(self, x):
        """
        Wrapper function for SHAP analysis.

        Args:
            x: Input tensor of shape (num_samples, num_features)

        Returns:
            torch.Tensor: Model embeddings (for gradient-based SHAP)
        """
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x.to(self.device)

        # Ensure gradients are enabled for SHAP gradient computation
        x_tensor.requires_grad_(True)

        # Process each sample individually
        embeddings = []
        for i in range(x_tensor.shape[0]):
            # Create a dictionary of modalities
            sample = x_tensor[i]

            # Split the sample into modalities based on the original dimensions
            modality_data = {}
            start_idx = 0
            for modality in self.modality_order:
                modality_dim = self.modality_dims[modality]
                modality_tensor = sample[start_idx:start_idx+modality_dim].unsqueeze(0)
                modality_tensor.requires_grad_(True)  # Enable gradients
                modality_data[modality] = modality_tensor
                start_idx += modality_dim

            # Forward pass through the model
            try:
                output = self.model(modality_data, self.gene_embeddings, self.edge_index_dict)
                patient_embedding = output['patient']
                embeddings.append(patient_embedding)
            except Exception as e:
                print(f"Error in IntegratedModelWrapper forward pass: {e}")
                # Return zeros as fallback
                fallback = torch.zeros((1, 128), device=self.device, requires_grad=True)
                embeddings.append(fallback)

        # Return tensor (not numpy) for gradient-based SHAP
        result = torch.cat(embeddings, dim=0)
        return result


class AutoencoderModelWrapper(nn.Module):
    """Wrapper for the autoencoder model for SHAP analysis."""
    
    def __init__(self, model, graph_node_features, graph_edge_index, graph_edge_weight):
        super().__init__()
        self.model = model
        self.graph_node_features = graph_node_features
        self.graph_edge_index = graph_edge_index
        self.graph_edge_weight = graph_edge_weight
        self.device = next(model.parameters()).device

    def forward(self, x):
        """
        Wrapper function for SHAP analysis.

        Args:
            x: Input tensor of shape (num_samples, num_features)

        Returns:
            torch.Tensor: Model embeddings (for gradient-based SHAP)
        """
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x.to(self.device)
            
        # Ensure gradients are enabled for SHAP gradient computation
        x_tensor.requires_grad_(True)

        # Process each sample
        embeddings = []
        for i in range(x_tensor.shape[0]):
            # Get the sample - reshape to match expected input format
            sample = x_tensor[i].unsqueeze(0)
            sample.requires_grad_(True)  # Enable gradients
            
            # The autoencoder expects input of shape (batch, genes, modalities)
            # Our input is concatenated, so we need to reshape it
            num_modalities = 4  # rnaseq, methylation, scnv, mirna
            num_genes = sample.shape[1] // num_modalities
            sample_reshaped = sample.view(1, num_genes, num_modalities)

            # Forward pass through the model
            try:
                # Get gene embeddings from the graph autoencoder
                mu, _, z_gene = self.model.graph_autoencoder.encode(
                    self.graph_node_features,
                    self.graph_edge_index,
                    edge_weight=self.graph_edge_weight
                )
                # Get patient embeddings from the omics processor
                patient_embedding = self.model.omics_processor.encode(sample_reshaped, z_gene)
                embeddings.append(patient_embedding)
            except Exception as e:
                print(f"Error in AutoencoderModelWrapper forward pass: {e}")
                # Return zeros as fallback
                fallback = torch.zeros((1, 128), device=self.device, requires_grad=True)
                embeddings.append(fallback)

        # Return tensor (not numpy) for gradient-based SHAP
        result = torch.cat(embeddings, dim=0)
        return result


class ProgressTracker:
    """
    Comprehensive progress tracking and state management for SHAP analysis.
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, "shap_analysis_progress.json")
        self.backup_dir = os.path.join(output_dir, "backups")
        self.state = self._load_or_create_state()
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def _load_or_create_state(self):
        """Load existing state or create new one."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"✓ Loaded existing progress state from {self.state_file}")
                return state
            except Exception as e:
                print(f"Warning: Could not load progress state: {e}")
                print("Creating new progress state...")
        
        # Create new state
        state = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "models": {
                "integrated": {
                    "shap_analysis_completed": False,
                    "shap_analysis_file": None,
                    "summary_plot_completed": False,
                    "modality_plots_completed": {},
                    "results_saved": False,
                    "fully_completed": False,
                    "last_error": None
                },
                "autoencoder": {
                    "shap_analysis_completed": False,
                    "shap_analysis_file": None,
                    "summary_plot_completed": False,
                    "modality_plots_completed": {},
                    "results_saved": False,
                    "fully_completed": False,
                    "last_error": None
                }
            },
            "combined_results_saved": False,
            "wandb_uploaded": False
        }
        
        # Assign state to self before calling _save_state
        self.state = state
        self._save_state()
        return state
    
    def _save_state(self):
        """Save current state to file."""
        # Defensive check - ensure state exists before trying to modify it
        if not hasattr(self, 'state') or self.state is None:
            print("Warning: _save_state called before state is initialized")
            return
            
        self.state["last_updated"] = datetime.now().isoformat()
        try:
            # Create backup first
            if os.path.exists(self.state_file):
                backup_file = os.path.join(self.backup_dir, f"progress_backup_{int(time.time())}.json")
                shutil.copy2(self.state_file, backup_file)
            
            # Save current state
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress state: {e}")
    
    def mark_shap_analysis_completed(self, model_name, shap_file_path):
        """Mark SHAP analysis as completed for a model."""
        self.state["models"][model_name]["shap_analysis_completed"] = True
        self.state["models"][model_name]["shap_analysis_file"] = shap_file_path
        self.state["models"][model_name]["last_error"] = None
        self._save_state()
        print(f"✓ Marked SHAP analysis completed for {model_name} model")
    
    def mark_summary_plot_completed(self, model_name):
        """Mark summary plot as completed for a model."""
        self.state["models"][model_name]["summary_plot_completed"] = True
        self._save_state()
        print(f"✓ Marked summary plot completed for {model_name} model")
    
    def mark_modality_plot_completed(self, model_name, modality):
        """Mark modality-specific plot as completed."""
        self.state["models"][model_name]["modality_plots_completed"][modality] = True
        self._save_state()
        print(f"✓ Marked {modality} plot completed for {model_name} model")
    
    def mark_results_saved(self, model_name):
        """Mark results as saved for a model."""
        self.state["models"][model_name]["results_saved"] = True
        self._save_state()
        print(f"✓ Marked results saved for {model_name} model")
    
    def mark_model_completed(self, model_name):
        """Mark entire model analysis as completed."""
        self.state["models"][model_name]["fully_completed"] = True
        self._save_state()
        print(f"✓ Marked {model_name} model fully completed")
    
    def mark_combined_results_saved(self):
        """Mark combined results as saved."""
        self.state["combined_results_saved"] = True
        self._save_state()
        print("✓ Marked combined results saved")
    
    def mark_wandb_uploaded(self):
        """Mark wandb upload as completed."""
        self.state["wandb_uploaded"] = True
        self._save_state()
        print("✓ Marked wandb upload completed")
    
    def record_error(self, model_name, error_msg):
        """Record an error for a model."""
        self.state["models"][model_name]["last_error"] = {
            "message": str(error_msg),
            "timestamp": datetime.now().isoformat()
        }
        self._save_state()
        print(f"⚠ Recorded error for {model_name} model: {error_msg}")
    
    def is_shap_analysis_completed(self, model_name):
        """Check if SHAP analysis is completed and valid."""
        if not self.state["models"][model_name]["shap_analysis_completed"]:
            return False
        
        # Verify the file still exists and is valid
        shap_file = self.state["models"][model_name]["shap_analysis_file"]
        if shap_file and os.path.exists(shap_file):
            try:
                # Try to load and validate the file
                data = joblib.load(shap_file)
                if 'shap_values' in data and data['shap_values'] is not None:
                    return True
                else:
                    print(f"⚠ SHAP file for {model_name} exists but is invalid")
                    return False
            except Exception as e:
                print(f"⚠ SHAP file for {model_name} exists but cannot be loaded: {e}")
                return False
        return False
    
    def is_summary_plot_completed(self, model_name):
        """Check if summary plot is completed."""
        if not self.state["models"][model_name]["summary_plot_completed"]:
            return False
        
        # Verify the file exists
        plot_file = os.path.join(self.output_dir, f"{model_name}_model_summary_plot.png")
        return os.path.exists(plot_file)
    
    def is_modality_plot_completed(self, model_name, modality):
        """Check if modality plot is completed."""
        if modality not in self.state["models"][model_name]["modality_plots_completed"]:
            return False
        
        if not self.state["models"][model_name]["modality_plots_completed"][modality]:
            return False
        
        # Verify the file exists
        plot_file = os.path.join(self.output_dir, f"{model_name}_model_{modality}_plot.png")
        return os.path.exists(plot_file)
    
    def is_results_saved(self, model_name):
        """Check if results are saved."""
        if not self.state["models"][model_name]["results_saved"]:
            return False
        
        # Verify the file exists
        results_file = os.path.join(self.output_dir, f"{model_name}_shap_results.joblib")
        return os.path.exists(results_file)
    
    def is_model_completed(self, model_name):
        """Check if entire model analysis is completed."""
        return self.state["models"][model_name]["fully_completed"]
    
    def get_incomplete_modalities(self, model_name, all_modalities):
        """Get list of modalities that still need plots."""
        incomplete = []
        for modality in all_modalities:
            if not self.is_modality_plot_completed(model_name, modality):
                incomplete.append(modality)
        return incomplete
    
    def print_status(self):
        """Print current analysis status."""
        print("\n" + "="*60)
        print("CURRENT ANALYSIS STATUS")
        print("="*60)
        
        for model_name in ["integrated", "autoencoder"]:
            model_state = self.state["models"][model_name]
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  SHAP Analysis: {'✓' if model_state['shap_analysis_completed'] else '✗'}")
            print(f"  Summary Plot: {'✓' if model_state['summary_plot_completed'] else '✗'}")
            
            modality_count = len([v for v in model_state['modality_plots_completed'].values() if v])
            total_modalities = len(model_state['modality_plots_completed'])
            print(f"  Modality Plots: {modality_count}/{total_modalities} completed")
            
            print(f"  Results Saved: {'✓' if model_state['results_saved'] else '✗'}")
            print(f"  Fully Completed: {'✓' if model_state['fully_completed'] else '✗'}")
            
            if model_state['last_error']:
                print(f"  Last Error: {model_state['last_error']['message']}")
        
        print(f"\nCombined Results: {'✓' if self.state['combined_results_saved'] else '✗'}")
        print(f"Wandb Upload: {'✓' if self.state['wandb_uploaded'] else '✗'}")
        print("="*60)


def safely_save_with_backup(data, file_path, backup_dir):
    """
    Safely save data with backup creation and atomic writing.
    
    Args:
        data: Data to save
        file_path: Target file path
        backup_dir: Directory for backups
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup of existing file
        if os.path.exists(file_path):
            timestamp = int(time.time())
            backup_name = f"{os.path.basename(file_path)}.backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(file_path, backup_path)
            print(f"  Created backup: {backup_path}")
        
        # Write to temporary file first (atomic operation)
        temp_path = file_path + ".tmp"
        joblib.dump(data, temp_path)
        
        # Move temporary file to final location
        shutil.move(temp_path, file_path)
        print(f"  ✓ Safely saved: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving {file_path}: {e}")
        # Clean up temporary file if it exists
        temp_path = file_path + ".tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def validate_shap_results(file_path):
    """
    Validate that SHAP results file is complete and valid.
    
    Args:
        file_path: Path to SHAP results file
        
    Returns:
        tuple: (is_valid, data_or_error_msg)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Load the data
        data = joblib.load(file_path)
        
        # Check required fields
        required_fields = ['shap_values', 'test_data', 'feature_names']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            if data[field] is None:
                return False, f"Field {field} is None"
        
        # Validate data shapes
        shap_values = data['shap_values']
        test_data = data['test_data']
        
        if len(shap_values.shape) != 2:
            return False, f"SHAP values should be 2D, got shape {shap_values.shape}"
        
        if len(test_data.shape) != 2:
            return False, f"Test data should be 2D, got shape {test_data.shape}"
        
        if shap_values.shape != test_data.shape:
            return False, f"SHAP values shape {shap_values.shape} != test data shape {test_data.shape}"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(shap_values)) or np.any(np.isinf(shap_values)):
            return False, "SHAP values contain NaN or infinite values"
        
        print(f"✓ Validated SHAP results: {file_path}")
        return True, data
        
    except Exception as e:
        return False, f"Error loading file: {str(e)}"


def robust_shap_analysis_with_retry(model_wrapper, background_data, test_data, model_name, 
                                  max_retries=3, reduce_samples_on_retry=True):
    """
    Perform SHAP analysis with retry logic and progressive sample reduction.
    
    Args:
        model_wrapper: Model wrapper for SHAP analysis
        background_data: Background data tensor
        test_data: Test data tensor
        model_name: Name of the model
        max_retries: Maximum number of retry attempts
        reduce_samples_on_retry: Whether to reduce sample size on retry
        
    Returns:
        tuple: (success, shap_values_or_error)
    """
    original_test_size = test_data.shape[0]
    original_bg_size = background_data.shape[0]
    current_test_data = test_data
    current_bg_data = background_data
    
    for attempt in range(max_retries + 1):
        try:
            print(f"\n{'='*20} SHAP Analysis Attempt {attempt + 1}/{max_retries + 1} {'='*20}")
            print(f"Background samples: {current_bg_data.shape[0]}, Test samples: {current_test_data.shape[0]}")
            
            # Try GradientExplainer first
            try:
                print(f"Trying GradientExplainer for {model_name} model...")
                explainer = shap.GradientExplainer(model_wrapper, current_bg_data)
                shap_values = explainer.shap_values(current_test_data)
                print(f"✓ GradientExplainer succeeded for {model_name} model!")
                return True, shap_values
                
            except Exception as e:
                print(f"GradientExplainer failed: {e}")
                
                # Try DeepExplainer
                try:
                    print(f"Trying DeepExplainer for {model_name} model...")
                    explainer = shap.DeepExplainer(model_wrapper, current_bg_data)
                    shap_values = explainer.shap_values(current_test_data)
                    print(f"✓ DeepExplainer succeeded for {model_name} model!")
                    return True, shap_values
                    
                except Exception as e2:
                    print(f"DeepExplainer failed: {e2}")
                    
                    # Try KernelExplainer as last resort
                    print(f"Trying KernelExplainer for {model_name} model (this will be slow)...")
                    bg_numpy = current_bg_data.cpu().numpy()
                    test_numpy = current_test_data.cpu().numpy()
                    explainer = shap.KernelExplainer(model_wrapper, bg_numpy)
                    shap_values = explainer.shap_values(test_numpy)
                    return True, shap_values
        
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries:
                if reduce_samples_on_retry:
                    # Reduce sample sizes for next attempt
                    new_test_size = max(5, current_test_data.shape[0] // 2)
                    new_bg_size = max(10, current_bg_data.shape[0] // 2)
                    
                    # Randomly sample reduced data
                    test_indices = torch.randperm(current_test_data.shape[0])[:new_test_size]
                    bg_indices = torch.randperm(current_bg_data.shape[0])[:new_bg_size]
                    
                    current_test_data = current_test_data[test_indices]
                    current_bg_data = current_bg_data[bg_indices]
                    
                    print(f"Reducing samples for retry: test={new_test_size}, background={new_bg_size}")
                
                # Force garbage collection and wait
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                time.sleep(2)
            else:
                print(f"✗ All attempts failed for {model_name} model")
                return False, str(e)
    
    return False, "Maximum retries exceeded"


def check_model_results_exist(args, model_name):
    """
    Check if SHAP results and visualizations already exist for a given model.
    
    Args:
        args: Command line arguments
        model_name: Name of the model ('integrated' or 'autoencoder')
        
    Returns:
        bool: True if all results exist, False otherwise
    """
    # Check for main result files
    shap_results_path = os.path.join(args.output_dir, f"{model_name}_shap_results.joblib")
    summary_plot_path = os.path.join(args.output_dir, f"{model_name}_model_summary_plot.png")
    
    if not (os.path.exists(shap_results_path) and os.path.exists(summary_plot_path)):
        return False
    
    # Check for modality-specific plots
    modalities = args.modalities.split(',')
    for modality in modalities:
        modality_plot_path = os.path.join(args.output_dir, f"{model_name}_model_{modality}_plot.png")
        if not os.path.exists(modality_plot_path):
            return False
    
    print(f"✓ All results for {model_name} model already exist. Skipping...")
    return True


def perform_shap_analysis_for_model(model_wrapper, background_data, test_data, model_name, 
                                  progress_tracker, backup_dir):
    """
    Perform SHAP analysis for a single model with comprehensive error handling.
    
    Args:
        model_wrapper: Model wrapper for SHAP analysis
        background_data: Background data tensor
        test_data: Test data tensor
        model_name: Name of the model ('integrated' or 'autoencoder')
        progress_tracker: ProgressTracker instance
        backup_dir: Directory for backups
        
    Returns:
        tuple: (success, shap_values_or_error)
    """
    print(f"\nRunning SHAP analysis for {model_name} model...")
    
    try:
        # Check if already completed
        if progress_tracker.is_shap_analysis_completed(model_name):
            print(f"✓ SHAP analysis for {model_name} already completed. Loading existing results...")
            shap_file = progress_tracker.state["models"][model_name]["shap_analysis_file"]
            is_valid, data_or_error = validate_shap_results(shap_file)
            if is_valid:
                return True, data_or_error['shap_values']
            else:
                print(f"⚠ Existing SHAP results invalid: {data_or_error}. Re-running analysis...")
        
        # Perform SHAP analysis with retry logic
        success, result = robust_shap_analysis_with_retry(
            model_wrapper, background_data, test_data, model_name
        )
        
        if success:
            # Save intermediate results immediately
            intermediate_file = os.path.join(backup_dir, f"{model_name}_shap_intermediate.joblib")
            intermediate_data = {
                'shap_values': result,
                'test_data': test_data.cpu().numpy(),
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name
            }
            
            if safely_save_with_backup(intermediate_data, intermediate_file, backup_dir):
                progress_tracker.mark_shap_analysis_completed(model_name, intermediate_file)
                print(f"✓ SHAP analysis completed and saved for {model_name} model")
                return True, result
            else:
                raise Exception("Failed to save intermediate SHAP results")
        else:
            progress_tracker.record_error(model_name, f"SHAP analysis failed: {result}")
            return False, result
            
    except Exception as e:
        error_msg = f"Error in SHAP analysis for {model_name}: {str(e)}"
        print(f"✗ {error_msg}")
        progress_tracker.record_error(model_name, error_msg)
        return False, error_msg


def generate_plots_for_model(shap_values, test_data, feature_names, modality_indices, 
                           model_name, args, progress_tracker):
    """
    Generate and save all plots for a single model with comprehensive error handling.
    
    Args:
        shap_values: SHAP values for the model
        test_data: Test data used for SHAP analysis
        feature_names: List of feature names
        modality_indices: Dictionary mapping modality names to feature indices
        model_name: Name of the model ('integrated' or 'autoencoder')
        args: Command line arguments
        progress_tracker: ProgressTracker instance
    """
    print(f"\nGenerating plots for {model_name} model...")
    
    try:
        # Convert tensors to numpy for plotting
        test_data_np = test_data.cpu().numpy() if isinstance(test_data, torch.Tensor) else test_data
        shap_values_np = shap_values
        
        # Handle case where SHAP values might be tensors
        if isinstance(shap_values, torch.Tensor):
            shap_values_np = shap_values.cpu().numpy()
        
        # Generate summary plot if not completed
        if not progress_tracker.is_summary_plot_completed(model_name):
            print(f"  Generating summary plot for {model_name} model...")
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_np, test_data_np, feature_names=feature_names,
                                  max_display=args.top_n_features, show=False)
                plt.title(f"Top {args.top_n_features} Features for {model_name.title()} Model")
                plt.tight_layout()
                summary_plot_path = os.path.join(args.output_dir, f"{model_name}_model_summary_plot.png")
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                progress_tracker.mark_summary_plot_completed(model_name)
                print(f"  ✓ Summary plot saved: {summary_plot_path}")
            except Exception as e:
                print(f"  ✗ Error generating summary plot: {e}")
                plt.close('all')  # Clean up any open figures
                raise e
        else:
            print(f"  ✓ Summary plot already completed for {model_name} model")
        
        # Generate modality-specific plots
        print(f"  Generating modality-specific plots for {model_name} model...")
        incomplete_modalities = progress_tracker.get_incomplete_modalities(model_name, list(modality_indices.keys()))
        
        for modality in incomplete_modalities:
            indices = modality_indices[modality]
            print(f"    Generating plot for {modality} modality...")
            try:
                plt.figure(figsize=(12, 8))
                modality_feature_names = [feature_names[i] for i in indices]
                modality_test_data = test_data_np[:, indices]
                modality_shap_values = shap_values_np[:, indices]

                shap.summary_plot(modality_shap_values, modality_test_data,
                                  feature_names=modality_feature_names,
                                  max_display=min(args.top_n_features, len(indices)),
                                  show=False)
                plt.title(f"Top Features for {modality.upper()} Modality ({model_name.title()} Model)")
                plt.tight_layout()
                modality_plot_path = os.path.join(args.output_dir, f"{model_name}_model_{modality}_plot.png")
                plt.savefig(modality_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                progress_tracker.mark_modality_plot_completed(model_name, modality)
                print(f"    ✓ {modality.title()} plot saved: {modality_plot_path}")
            except Exception as e:
                print(f"    ✗ Error generating {modality} plot: {e}")
                plt.close('all')  # Clean up any open figures
                # Continue with other modalities rather than failing completely
                continue
        
        # Check if any modalities were skipped due to errors
        remaining_incomplete = progress_tracker.get_incomplete_modalities(model_name, list(modality_indices.keys()))
        if remaining_incomplete:
            print(f"  ⚠ Some modality plots failed: {remaining_incomplete}")
        else:
            print(f"  ✓ All modality plots completed for {model_name} model")
        
        # Save individual model results if not already saved
        if not progress_tracker.is_results_saved(model_name):
            print(f"  Saving results for {model_name} model...")
            model_results = {
                'shap_values': shap_values_np,
                'test_data': test_data_np,
                'feature_names': feature_names,
                'modality_indices': modality_indices,
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name
            }
            
            model_results_path = os.path.join(args.output_dir, f"{model_name}_shap_results.joblib")
            backup_dir = os.path.join(args.output_dir, "backups")
            
            if safely_save_with_backup(model_results, model_results_path, backup_dir):
                progress_tracker.mark_results_saved(model_name)
                print(f"  ✓ {model_name.title()} model results saved: {model_results_path}")
            else:
                raise Exception(f"Failed to save results for {model_name} model")
        else:
            print(f"  ✓ Results already saved for {model_name} model")
        
        # Force garbage collection to free memory
        del shap_values_np, test_data_np
        if 'modality_test_data' in locals():
            del modality_test_data
        if 'modality_shap_values' in locals():
            del modality_shap_values
        gc.collect()
        
        print(f"✓ All plots and results for {model_name} model completed!")
        
    except Exception as e:
        error_msg = f"Error generating plots for {model_name}: {str(e)}"
        print(f"✗ {error_msg}")
        progress_tracker.record_error(model_name, error_msg)
        # Clean up any open figures
        plt.close('all')
        raise e


def test_shap_analysis_minimal(args, models_and_data):
    """
    Test SHAP analysis with minimal samples (1 background, 1 test) to ensure functionality.
    
    Args:
        args: Command line arguments
        models_and_data: Tuple containing loaded models and data
        
    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "="*80)
    print("RUNNING SHAP ANALYSIS TEST (1 SAMPLE)")
    print("="*80)
    
    try:
        (integrated_wrapper, autoencoder_wrapper, raw_omics_data_dict, 
         feature_names, modality_indices) = models_and_data
        
        # Prepare minimal test data
        integrated_data = []
        for modality, tensor in raw_omics_data_dict.items():
            integrated_data.append(tensor.cpu().numpy())
        integrated_data = np.concatenate(integrated_data, axis=1)
        
        # Use just 1 sample for background and 1 for test
        if integrated_data.shape[0] < 2:
            print("⚠ Not enough samples for test. Need at least 2 samples.")
            return False
            
        background_data = torch.tensor(integrated_data[0:1], dtype=torch.float32, device=torch.device(args.device))
        test_data = torch.tensor(integrated_data[1:2], dtype=torch.float32, device=torch.device(args.device))
        
        print(f"Testing with {background_data.shape[0]} background sample and {test_data.shape[0]} test sample")
        
        # Test integrated model
        print("  Testing integrated model wrapper...")
        try:
            integrated_output = integrated_wrapper(test_data)
            print(f"  ✓ Integrated model wrapper works. Output shape: {integrated_output.shape}")
        except Exception as e:
            print(f"  ✗ Integrated model wrapper failed: {e}")
            return False
        
        # Test autoencoder model  
        print("  Testing autoencoder model wrapper...")
        try:
            autoencoder_output = autoencoder_wrapper(test_data)
            print(f"  ✓ Autoencoder model wrapper works. Output shape: {autoencoder_output.shape}")
        except Exception as e:
            print(f"  ✗ Autoencoder model wrapper failed: {e}")
            return False
        
        # Test minimal SHAP analysis on integrated model
        print("  Testing SHAP analysis on integrated model...")
        try:
            success, result = robust_shap_analysis_with_retry(
                integrated_wrapper, background_data, test_data, 'integrated_test', 
                max_retries=1, reduce_samples_on_retry=False
            )
            if success:
                print(f"  ✓ SHAP analysis works on integrated model. SHAP values shape: {result.shape}")
            else:
                print(f"  ✗ SHAP analysis failed on integrated model: {result}")
                return False
        except Exception as e:
            print(f"  ✗ SHAP analysis error on integrated model: {e}")
            return False
        
        # Test minimal SHAP analysis on autoencoder model
        print("  Testing SHAP analysis on autoencoder model...")
        try:
            success, result = robust_shap_analysis_with_retry(
                autoencoder_wrapper, background_data, test_data, 'autoencoder_test',
                max_retries=1, reduce_samples_on_retry=False
            )
            if success:
                print(f"  ✓ SHAP analysis works on autoencoder model. SHAP values shape: {result.shape}")
            else:
                print(f"  ✗ SHAP analysis failed on autoencoder model: {result}")
                return False
        except Exception as e:
            print(f"  ✗ SHAP analysis error on autoencoder model: {e}")
            return False
        
        print("✓ SHAP analysis test PASSED - all components working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ SHAP analysis test FAILED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def load_gene_names(data_path, cancer_type):
    """
    Load gene names from the prepared data file.

    Args:
        data_path: Path to the prepared data joblib file
        cancer_type: Cancer type to analyze

    Returns:
        dict: Dictionary mapping modality to gene names
    """
    # Load prepared data
    prepared_data = joblib.load(data_path)
    if cancer_type not in prepared_data:
        print(f"Error: Could not find data for {cancer_type} in {data_path}")
        return None

    # Extract data
    cancer_data = prepared_data[cancer_type]

    # Extract omics data
    omics_data = cancer_data['omics_data']

    # Create dictionary mapping modality to gene names
    gene_names = {}
    for modality, df in omics_data.items():
        if modality in ['clinical']:
            continue

        # Get gene names from DataFrame columns (excluding patient_id)
        if 'patient_id' in df.columns:
            gene_names[modality] = df.columns.drop('patient_id').tolist()
        else:
            gene_names[modality] = df.columns.tolist()

    return gene_names


def map_feature_indices_to_genes(shap_results, data_path, cancer_type, output_dir):
    """
    Map feature indices to gene names and create more interpretable visualizations.

    Args:
        shap_results: Dictionary containing SHAP analysis results
        data_path: Path to the prepared data joblib file
        cancer_type: Cancer type to analyze
        output_dir: Directory to save the results

    Returns:
        tuple: (feature_to_gene dict, gene_feature_names list)
    """
    # Load gene names
    gene_names = load_gene_names(data_path, cancer_type)
    if gene_names is None:
        return None

    # Create output directory and ensure parent directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract feature names and modality indices from SHAP results
    feature_names = shap_results['integrated_model']['feature_names']
    modality_indices = shap_results['integrated_model']['modality_indices']

    # Create mapping from feature indices to gene names
    feature_to_gene = {}
    for modality, indices in modality_indices.items():
        if modality not in gene_names:
            continue

        modality_gene_names = gene_names[modality]
        for i, idx in enumerate(indices):
            if i < len(modality_gene_names):
                feature_to_gene[idx] = f"{modality}_{modality_gene_names[i]}"
            else:
                feature_to_gene[idx] = f"{modality}_{i}"

    # Create new feature names with gene names
    gene_feature_names = []
    for i, name in enumerate(feature_names):
        if i in feature_to_gene:
            gene_feature_names.append(feature_to_gene[i])
        else:
            gene_feature_names.append(name)

    # Save mapping to file
    mapping_df = pd.DataFrame({
        'feature_index': list(range(len(feature_names))),
        'feature_name': feature_names,
        'gene_name': gene_feature_names
    })
    mapping_df.to_csv(os.path.join(output_dir, "feature_to_gene_mapping.csv"), index=False)

    return feature_to_gene, gene_feature_names


def test_gene_mapper_functionality(args):
    """
    Test gene mapper functionality with dummy SHAP results.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "="*80)
    print("RUNNING GENE MAPPER TEST")
    print("="*80)
    
    try:
        print("  Testing gene name loading...")
        # Test gene name loading using local function
        gene_names = load_gene_names(args.data_path, args.cancer_type)
        if gene_names is None:
            print("  ✗ Failed to load gene names")
            return False
        
        print(f"  ✓ Gene names loaded successfully. Found modalities: {list(gene_names.keys())}")
        for modality, names in gene_names.items():
            print(f"    {modality}: {len(names)} genes")
        
        # Create dummy SHAP results for testing
        print("  Creating dummy SHAP results for testing...")
        dummy_feature_names = []
        dummy_modality_indices = {}
        start_idx = 0
        
        for modality, names in gene_names.items():
            num_features = min(10, len(names))  # Use max 10 features per modality for test
            for i in range(num_features):
                dummy_feature_names.append(f"{modality}_{i}")
            dummy_modality_indices[modality] = list(range(start_idx, start_idx + num_features))
            start_idx += num_features
        
        dummy_shap_results = {
            'integrated_model': {
                'feature_names': dummy_feature_names,
                'modality_indices': dummy_modality_indices
            }
        }
        
        print(f"  Created dummy results with {len(dummy_feature_names)} features")
        
        # Test mapping functionality using local function
        print("  Testing feature-to-gene mapping...")
        test_output_dir = os.path.join(args.output_dir, "gene_mapper_test")
        
        result = map_feature_indices_to_genes(
            dummy_shap_results, args.data_path, args.cancer_type, test_output_dir
        )
        
        if result is None:
            print("  ✗ Failed to create feature-to-gene mapping")
            return False
        
        # Unpack the result tuple
        feature_to_gene, gene_feature_names = result
        
        print(f"  ✓ Feature-to-gene mapping created successfully")
        print(f"    Mapped {len(feature_to_gene)} features to genes")
        print(f"    Generated {len(gene_feature_names)} gene feature names")
        
        # Check if mapping file was created
        mapping_file = os.path.join(test_output_dir, "feature_to_gene_mapping.csv")
        if os.path.exists(mapping_file):
            print(f"  ✓ Mapping file created: {mapping_file}")
        else:
            print(f"  ✗ Mapping file not created")
            return False
        
        print("✓ Gene mapper test PASSED - functionality working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Gene mapper test FAILED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def run_pre_analysis_tests(args):
    """
    Run all pre-analysis tests to ensure SHAP analysis and gene mapper work correctly.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (test_passed, models_and_data_or_None)
    """
    print("\n" + "="*80)
    print("RUNNING PRE-ANALYSIS TESTS")
    print("="*80)
    print("Testing with minimal samples to ensure functionality before full analysis...")
    
    try:
        # Load configurations
        print("Loading model configurations...")
        integrated_config = load_config(args.integrated_config_path)
        autoencoder_config = load_config(args.autoencoder_config_path)

        # Load prepared data
        print(f"Loading prepared data from {args.data_path}...")
        prepared_data = load_prepared_data(args.data_path)
        if not prepared_data or args.cancer_type not in prepared_data:
            raise Exception(f"Could not load or find data for {args.cancer_type} in {args.data_path}")

        # Load models (same as main analysis)
        print("Loading integrated transformer GCN model...")
        integrated_result = load_integrated_model(args, integrated_config, prepared_data)
        if integrated_result is None:
            raise Exception("Failed to load integrated model")

        integrated_model, gene_embeddings, edge_index_dict, raw_omics_data_dict, patient_ids, gene_list = integrated_result

        print("Loading joint autoencoder model...")
        autoencoder_result = load_autoencoder_model(args, autoencoder_config, prepared_data)
        if autoencoder_result is None:
            raise Exception("Failed to load autoencoder model")

        autoencoder_model, graph_node_features, graph_edge_index, graph_edge_weight, omics_data_dict, ae_patient_ids, ae_gene_list = autoencoder_result

        # Create model wrappers
        print("Creating model wrappers...")
        integrated_wrapper = IntegratedModelWrapper(integrated_model, gene_embeddings, edge_index_dict, raw_omics_data_dict)
        autoencoder_wrapper = AutoencoderModelWrapper(autoencoder_model, graph_node_features, graph_edge_index, graph_edge_weight)

        # Create feature names and modality indices
        feature_names = []
        for modality, tensor in raw_omics_data_dict.items():
            for i in range(tensor.shape[1]):
                feature_names.append(f"{modality}_{i}")

        modality_indices = {}
        start_idx = 0
        for modality, tensor in raw_omics_data_dict.items():
            modality_dim = tensor.shape[1]
            modality_indices[modality] = list(range(start_idx, start_idx + modality_dim))
            start_idx += modality_dim

        models_and_data = (integrated_wrapper, autoencoder_wrapper, raw_omics_data_dict, 
                          feature_names, modality_indices)

        # Test 1: SHAP analysis functionality
        if not test_shap_analysis_minimal(args, models_and_data):
            print("✗ SHAP analysis test failed. Cannot proceed with full analysis.")
            return False, None

        # Test 2: Gene mapper functionality  
        if not test_gene_mapper_functionality(args):
            print("✗ Gene mapper test failed. Cannot proceed with full analysis.")
            return False, None

        print("\n" + "="*80)
        print("✓ ALL PRE-ANALYSIS TESTS PASSED!")
        print("="*80)
        print("Proceeding with full SHAP analysis...")
        
        return True, models_and_data

    except Exception as e:
        print(f"\n✗ PRE-ANALYSIS TESTS FAILED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False, None


def main():
    """Main function to run SHAP analysis."""
    # Parse arguments
    args = parse_args()

    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "gene_level"), exist_ok=True)

    # RUN PRE-ANALYSIS TESTS FIRST
    test_passed, test_models_and_data = run_pre_analysis_tests(args)
    if not test_passed:
        print("\n✗ Pre-analysis tests failed. Exiting.")
        sys.exit(1)

    # Extract models and data from test results to reuse
    (integrated_wrapper, autoencoder_wrapper, raw_omics_data_dict, 
     test_feature_names, test_modality_indices) = test_models_and_data

    # Initialize progress tracker after tests pass
    progress_tracker = ProgressTracker(args.output_dir)
    progress_tracker.print_status()

    # Initialize wandb for experiment tracking
    run = initialize_wandb(args)

    print(f"\nStarting full SHAP analysis for {args.cancer_type} cancer...")
    print(f"Using device: {args.device}")

    try:
        # Prepare data for SHAP analysis
        print("Preparing data for SHAP analysis...")

        # Convert raw omics data to tensors for gradient-based SHAP
        integrated_data = []
        for modality, tensor in raw_omics_data_dict.items():
            integrated_data.append(tensor.cpu().numpy())

        # Concatenate all modalities
        integrated_data = np.concatenate(integrated_data, axis=1)
        
        print(f"Total available samples: {integrated_data.shape[0]}")

        # Create background data for SHAP (convert to tensors) - limit to 50 samples max
        max_background_samples = min(50, args.num_background_samples, integrated_data.shape[0] // 2)
        background_indices = np.random.choice(integrated_data.shape[0],
                                             size=max_background_samples,
                                             replace=False)
        background_data = torch.tensor(integrated_data[background_indices], 
                                     dtype=torch.float32, device=torch.device(args.device))

        # Create test data for SHAP (convert to tensors) - limit to 100 samples max
        remaining_indices = [i for i in range(integrated_data.shape[0]) if i not in background_indices]
        max_test_samples = min(100, args.num_test_samples, len(remaining_indices))
        test_indices = np.random.choice(remaining_indices, size=max_test_samples, replace=False)
        test_data = torch.tensor(integrated_data[test_indices], 
                               dtype=torch.float32, device=torch.device(args.device))
        
        print(f"Using {len(background_indices)} background samples and {len(test_indices)} test samples for SHAP analysis")
        print(f"This should complete much faster with the reduced sample size!")

        # Use feature names and modality indices from test results
        feature_names = test_feature_names
        modality_indices = test_modality_indices

        # Initialize modality plots in progress tracker
        modalities = list(modality_indices.keys())
        for model_name in ['integrated', 'autoencoder']:
            for modality in modalities:
                if modality not in progress_tracker.state["models"][model_name]["modality_plots_completed"]:
                    progress_tracker.state["models"][model_name]["modality_plots_completed"][modality] = False
        progress_tracker._save_state()

        # Sequential SHAP analysis and visualization for each model
        
        # Initialize results dictionary
        shap_results = {
            'integrated_model': {},
            'autoencoder_model': {},
            'test_indices': test_indices,
            'background_indices': background_indices,
            'feature_names': feature_names,
            'modality_indices': modality_indices
        }
        
        # 1. Process Integrated Model
        print("\n" + "="*80)
        print("PROCESSING INTEGRATED MODEL")
        print("="*80)
        
        try:
            if not progress_tracker.is_model_completed('integrated'):
                # Perform SHAP analysis for integrated model
                success, result = perform_shap_analysis_for_model(
                    integrated_wrapper, background_data, test_data, 'integrated', 
                    progress_tracker, progress_tracker.backup_dir
                )
                
                if success:
                    integrated_shap_values = result
                    
                    # Generate and save all plots for integrated model
                    generate_plots_for_model(
                        integrated_shap_values, test_data, feature_names, modality_indices, 
                        'integrated', args, progress_tracker
                    )
                    
                    # Mark model as completed
                    progress_tracker.mark_model_completed('integrated')
                    
                    # Store results
                    shap_results['integrated_model'] = {
                        'shap_values': integrated_shap_values,
                        'test_data': test_data.cpu().numpy(),
                        'test_indices': test_indices,
                        'background_indices': background_indices,
                        'feature_names': feature_names,
                        'modality_indices': modality_indices
                    }
                else:
                    raise Exception(f"SHAP analysis failed for integrated model: {result}")
            else:
                # Load existing results
                integrated_results_path = os.path.join(args.output_dir, "integrated_shap_results.joblib")
                if os.path.exists(integrated_results_path):
                    loaded_results = joblib.load(integrated_results_path)
                    shap_results['integrated_model'] = loaded_results
                    print("✓ Loaded existing integrated model results")
                else:
                    print("⚠ Integrated model marked as completed but results file missing. Re-running...")
                    progress_tracker.state["models"]["integrated"]["fully_completed"] = False
                    progress_tracker._save_state()
                    raise Exception("Results file missing, need to re-run")
        
        except Exception as e:
            print(f"✗ Error processing integrated model: {e}")
            progress_tracker.record_error('integrated', str(e))
            print("Continuing with autoencoder model...")
        
        # 2. Process Autoencoder Model
        print("\n" + "="*80)
        print("PROCESSING AUTOENCODER MODEL")
        print("="*80)
        
        try:
            if not progress_tracker.is_model_completed('autoencoder'):
                # Perform SHAP analysis for autoencoder model
                success, result = perform_shap_analysis_for_model(
                    autoencoder_wrapper, background_data, test_data, 'autoencoder',
                    progress_tracker, progress_tracker.backup_dir
                )
                
                if success:
                    autoencoder_shap_values = result
                    
                    # Generate and save all plots for autoencoder model
                    generate_plots_for_model(
                        autoencoder_shap_values, test_data, feature_names, modality_indices, 
                        'autoencoder', args, progress_tracker
                    )
                    
                    # Mark model as completed
                    progress_tracker.mark_model_completed('autoencoder')
                    
                    # Store results
                    shap_results['autoencoder_model'] = {
                        'shap_values': autoencoder_shap_values,
                        'test_data': test_data.cpu().numpy(),
                        'test_indices': test_indices,
                        'background_indices': background_indices,
                        'feature_names': feature_names,
                        'modality_indices': modality_indices
                    }
                else:
                    raise Exception(f"SHAP analysis failed for autoencoder model: {result}")
            else:
                # Load existing results
                autoencoder_results_path = os.path.join(args.output_dir, "autoencoder_shap_results.joblib")
                if os.path.exists(autoencoder_results_path):
                    loaded_results = joblib.load(autoencoder_results_path)
                    shap_results['autoencoder_model'] = loaded_results
                    print("✓ Loaded existing autoencoder model results")
                else:
                    print("⚠ Autoencoder model marked as completed but results file missing. Re-running...")
                    progress_tracker.state["models"]["autoencoder"]["fully_completed"] = False
                    progress_tracker._save_state()
                    raise Exception("Results file missing, need to re-run")
        
        except Exception as e:
            print(f"✗ Error processing autoencoder model: {e}")
            progress_tracker.record_error('autoencoder', str(e))
            print("Continuing with result combination...")

        # Save combined results
        print("\n" + "="*80)
        print("SAVING COMBINED RESULTS")
        print("="*80)
        
        if not progress_tracker.state["combined_results_saved"]:
            try:
                # Ensure output directory exists
                os.makedirs(args.output_dir, exist_ok=True)

                # Save combined results
                results_path = os.path.join(args.output_dir, "shap_results.joblib")
                backup_dir = os.path.join(args.output_dir, "backups")
                
                if safely_save_with_backup(shap_results, results_path, backup_dir):
                    progress_tracker.mark_combined_results_saved()
                    print(f"✓ Combined SHAP results saved to {results_path}")
                else:
                    raise Exception("Failed to save combined results")
            except Exception as e:
                print(f"✗ Error saving combined results: {e}")
                progress_tracker.record_error('combined', str(e))
        else:
            print("✓ Combined results already saved")

        # Upload SHAP results to wandb
        if not progress_tracker.state["wandb_uploaded"]:
            try:
                print("\n" + "="*80)
                print("UPLOADING TO WANDB")
                print("="*80)
                
                upload_shap_results_to_wandb(args, shap_results, modality_indices)
                progress_tracker.mark_wandb_uploaded()
                print("✓ Successfully uploaded to Weights and Biases")
            except Exception as e:
                print(f"⚠ Error uploading to wandb: {e}")
                print("Continuing without wandb upload...")
        else:
            print("✓ Wandb upload already completed")

        # Final status
        print("\n" + "="*80)
        print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"All results saved to {args.output_dir}")
        progress_tracker.print_status()

    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"\nProgress has been saved. You can resume by running the script again.")
        if 'progress_tracker' in locals():
            progress_tracker.print_status()
        
        # Re-raise the exception to ensure proper exit code
        raise e
    
    finally:
        # Finish the wandb run
        try:
            wandb.finish()
            print("Weights and Biases run completed!")
        except:
            pass
        
        # Clean up any remaining matplotlib figures
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return

if __name__ == "__main__":
    main()

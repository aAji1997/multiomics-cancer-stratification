import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import time
import random

# Local imports
from model import JointAutoencoder
from data_utils import load_prepared_data, JointOmicsDataset, prepare_graph_data

def test_joint_autoencoder(args):
    """Tests the Joint Autoencoder on a small subset of genes."""

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Loading full data from: {args.data_path}")
    prepared_data = load_prepared_data(args.data_path)
    if not prepared_data or args.cancer_type not in prepared_data:
        print(f"Error: Could not load or find data for {args.cancer_type} in {args.data_path}")
        return

    cancer_data = prepared_data[args.cancer_type]
    full_omics_data_dict = cancer_data['omics_data']
    full_adj_matrix = cancer_data['adj_matrix']
    full_gene_list = cancer_data['gene_list']
    num_full_genes = len(full_gene_list)
    print(f"Full dataset loaded: {num_full_genes} genes.")

    # --- Subsetting Genes ---
    if args.num_subset_genes >= num_full_genes:
        print("Warning: num_subset_genes is >= total genes. Using all genes.")
        args.num_subset_genes = num_full_genes
        subset_indices = list(range(num_full_genes))
    else:
        print(f"Selecting a random subset of {args.num_subset_genes} genes.")
        subset_indices = sorted(random.sample(range(num_full_genes), args.num_subset_genes))

    subset_gene_list = [full_gene_list[i] for i in subset_indices]
    num_genes = len(subset_gene_list)

    # Subset Adjacency Matrix
    subset_adj_matrix = full_adj_matrix[np.ix_(subset_indices, subset_indices)]
    print(f"Subset adjacency matrix shape: {subset_adj_matrix.shape}")

    # Subset Omics Data
    subset_omics_data_dict = {}
    modalities_to_use = args.modalities.split(',') if args.modalities else ['rnaseq', 'methylation', 'scnv', 'miRNA']
    print(f"Subsetting omics data for modalities: {modalities_to_use}")
    for omics_type in modalities_to_use:
        if omics_type in full_omics_data_dict:
            full_df = full_omics_data_dict[omics_type]
            # Ensure patient_id is a column for easy access
            if full_df.index.name == 'patient_id':
                 full_df = full_df.reset_index()

            # Select patient_id column + subset gene columns
            columns_to_keep = ['patient_id'] + subset_gene_list
            # Filter columns that actually exist in the dataframe
            existing_cols_to_keep = [col for col in columns_to_keep if col in full_df.columns]
            if len(existing_cols_to_keep) < len(columns_to_keep):
                 missing_subset_genes = set(columns_to_keep) - set(existing_cols_to_keep)
                 print(f"  Warning: Missing subset genes in {omics_type}: {missing_subset_genes}")

            subset_omics_data_dict[omics_type] = full_df[existing_cols_to_keep].copy()
            print(f"  Subset {omics_type} shape: {subset_omics_data_dict[omics_type].shape}")
        else:
            print(f"  Warning: Modality '{omics_type}' not found in full omics data.")

    # --- Prepare Subset Data ---
    print("Preparing subset graph data...")
    graph_node_features, graph_edge_index, graph_adj_tensor = prepare_graph_data(subset_adj_matrix, use_identity_features=True)
    graph_feature_dim = graph_node_features.shape[1]

    graph_node_features = graph_node_features.to(device)
    graph_edge_index = graph_edge_index.to(device)
    graph_adj_tensor = graph_adj_tensor.to(device)

    print("Creating JointOmicsDataset with subset data...")
    try:
        joint_omics_dataset = JointOmicsDataset(subset_omics_data_dict, subset_gene_list, modalities=modalities_to_use)
    except (ValueError, KeyError) as e:
        print(f"Error creating JointOmicsDataset: {e}")
        return

    if len(joint_omics_dataset) == 0:
        print("Error: JointOmicsDataset is empty after subsetting/processing.")
        return

    num_modalities = joint_omics_dataset.num_modalities
    # Use a smaller batch size for testing if needed
    test_batch_size = min(args.batch_size, 4) if args.batch_size else 4
    dataloader = DataLoader(joint_omics_dataset, batch_size=test_batch_size, shuffle=True)
    print(f"Created DataLoader with {len(dataloader)} batches (batch size: {test_batch_size}).")

    # --- Model Instantiation (using subset dimensions) ---
    # Use smaller embedding dims for quick test
    test_gene_emb_dim = min(args.gene_embedding_dim, 32)
    test_patient_emb_dim = min(args.patient_embedding_dim, 64)
    model = JointAutoencoder(
        num_nodes=num_genes,
        num_modalities=num_modalities,
        graph_feature_dim=graph_feature_dim,
        gene_embedding_dim=test_gene_emb_dim,
        patient_embedding_dim=test_patient_emb_dim,
        graph_dropout=args.graph_dropout
    ).to(device)

    print("\nTest Model Architecture:")
    # print(model) # Can be verbose
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters (Subset Model): {total_params:,}")

    # --- Loss and Optimizer ---
    omics_loss_fn = F.mse_loss
    graph_loss_fn = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Mini Training Loop ---
    print(f"\nStarting test training loop ({args.test_epochs} epochs, max {args.test_batches} batches/epoch)...")
    start_time_total = time.time()
    model.train()
    for epoch in range(args.test_epochs):
        epoch_start_time = time.time()
        batch_count = 0
        for batch_idx, omics_batch_structured in enumerate(dataloader):
            if args.test_batches is not None and batch_count >= args.test_batches:
                 break # Limit batches per epoch for testing

            omics_batch_structured = omics_batch_structured.to(device)
            omics_reconstructed, adj_reconstructed, z_patient, z_gene = model(
                graph_node_features, graph_edge_index, omics_batch_structured
            )

            loss_o = omics_loss_fn(omics_reconstructed, omics_batch_structured)
            loss_g = graph_loss_fn(adj_reconstructed, graph_adj_tensor)
            combined_loss = args.omics_loss_weight * loss_o + args.graph_loss_weight * loss_g

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0: # Print loss occasionally within epoch
                 print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {combined_loss.item():.6f}")
            batch_count += 1

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{args.test_epochs}] completed in {epoch_duration:.2f} sec")

    total_training_time = time.time() - start_time_total
    print(f"\nTest training finished. Total duration: {total_training_time:.2f} sec")
    print("If no errors occurred, the architecture and data flow seem okay.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Joint Graph-Omics Autoencoder on Subset')

    # Data/Model Params (Keep relevant ones, fix others)
    parser.add_argument('--data_path', type=str, default='data/prepared_data_both.joblib',
                        help='Path to the prepared data joblib file relative to workspace root')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type to test on')
    parser.add_argument('--modalities', type=str, default='rnaseq,methylation,scnv,miRNA',
                        help='Comma-separated list of omics modalities to use')
    parser.add_argument('--num_subset_genes', type=int, default=500,
                        help='Number of random genes to subset for testing')
    # Use smaller default dims for testing
    parser.add_argument('--gene_embedding_dim', type=int, default=32,
                        help='Dimension of the latent gene embeddings (Z_gene) for test')
    parser.add_argument('--patient_embedding_dim', type=int, default=64,
                        help='Dimension of the latent patient embeddings (z_p) for test')
    parser.add_argument('--graph_dropout', type=float, default=0.1, # Lower dropout for test
                        help='Dropout rate for GCN layers in graph AE')

    # Training Params (Fixed/Simplified for testing)
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, # Small batch size for testing
                        help='Batch size for testing')
    parser.add_argument('--omics_loss_weight', type=float, default=1.0,
                        help='Weight for the omics reconstruction loss')
    parser.add_argument('--graph_loss_weight', type=float, default=0.5,
                        help='Weight for the graph reconstruction loss')
    parser.add_argument('--test_epochs', type=int, default=2, # Fixed small number of epochs
                        help='Number of epochs for the test run')
    parser.add_argument('--test_batches', type=int, default=10, # Limit batches per epoch
                        help='Maximum number of batches to run per epoch for testing (None for all)')

    args = parser.parse_args()

    test_joint_autoencoder(args) 
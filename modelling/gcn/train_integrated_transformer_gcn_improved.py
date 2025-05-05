# modelling/gcn/train_integrated_transformer_gcn_improved.py
# Improved training script with batch-based modality cycling and gradient accumulation

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import joblib
import numpy as np
import pandas as pd
import argparse
import os
import sys
import time
import gc
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import scipy.sparse as sp

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute import paths
from modelling.gcn.train_risk_stratification_gcn import contrastive_loss, create_patient_gene_edges
from modelling.gcn.train_integrated_transformer_gcn import (
    load_raw_omics_data, load_gene_embeddings, load_gene_interactions_and_masks,
    augment_raw_omics, augment_edges, save_checkpoint, load_checkpoint,
    structure_preservation_loss, gene_interaction_loss
)
from modelling.gcn.risk_stratification_loss import risk_stratification_loss

import wandb
from modelling.gcn.integrated_transformer_gcn_model import IntegratedTransformerGCN
# --- Improved Training Functions --- #

def train_with_joint_modalities(
    model, optimizer, raw_omics_data_dict, gene_embeddings, edge_index_dict,
    graph_adj_tensor, gene_masks, original_gene_embeddings, args, device, scaler=None
):
    """
    Improved training function that processes all modalities together in each batch
    and uses gradient accumulation for memory efficiency.

    Args:
        model: The IntegratedTransformerGCN model
        optimizer: The optimizer
        raw_omics_data_dict: Dictionary of raw omics data tensors
        gene_embeddings: Gene embeddings tensor
        edge_index_dict: Dictionary of edge indices
        graph_adj_tensor: Graph adjacency tensor for gene-gene interactions
        gene_masks: Dictionary of gene masks
        original_gene_embeddings: Original gene embeddings for structure preservation
        args: Training arguments
        device: Device to train on
        scaler: Optional GradScaler for mixed precision training

    Returns:
        Dictionary of training metrics
    """
    model.train()

    # Get modality order
    modality_order = sorted(raw_omics_data_dict.keys())

    # Get number of patients
    num_patients = next(iter(raw_omics_data_dict.values())).shape[0]

    # Calculate number of batches
    if args.patient_batch_size <= 0:
        args.patient_batch_size = 32  # Default to 32 if invalid
        print(f"Invalid batch size. Using default batch size of {args.patient_batch_size}")

    num_batches = (num_patients + args.patient_batch_size - 1) // args.patient_batch_size

    # Initialize loss tracking
    total_loss = 0.0
    contrastive_loss_sum = 0.0
    graph_loss_sum = 0.0
    omics_loss_sum = 0.0
    structure_loss_sum = 0.0
    gene_interaction_loss_sum = 0.0
    risk_stratification_loss_sum = 0.0

    # Track problematic modalities to skip them after multiple failures
    if not hasattr(model, 'problematic_modalities'):
        model.problematic_modalities = {}

    # Reset error counter for modalities at the start of each epoch
    # This is a new epoch, so reset the counter
    model.problematic_modalities = {}

    # Create batch progress bar
    batch_pbar = tqdm(total=num_batches, desc=f'Batches', leave=False)

    # Shuffle patient indices
    patient_indices = torch.randperm(num_patients).tolist()

    # Initialize graph loss function if needed
    graph_loss_fn = None
    if graph_adj_tensor is not None:
        # Calculate BCE pos_weight for graph loss
        num_nodes = graph_adj_tensor.shape[0]
        num_possible_edges = num_nodes * num_nodes
        num_positives = torch.sum(graph_adj_tensor).item()
        num_negatives = num_possible_edges - num_positives
        if num_positives == 0:
            base_pos_weight = 1.0
        else:
            base_pos_weight = num_negatives / num_positives
        final_pos_weight = base_pos_weight * 1.0  # Adjust factor if needed
        pos_weight_tensor = torch.tensor([final_pos_weight], device=device)
        graph_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Define gradient accumulation steps (smaller for larger batches)
    grad_accum_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1
    if grad_accum_steps <= 0:
        grad_accum_steps = 1
        print(f"Invalid gradient accumulation steps. Using default value of {grad_accum_steps}")

    # Process patients in mini-batches
    for batch_idx in range(num_batches):
        # Get batch patient indices
        start_idx = batch_idx * args.patient_batch_size
        end_idx = min(start_idx + args.patient_batch_size, num_patients)
        batch_patient_indices = patient_indices[start_idx:end_idx]

        # Extract batch data for ALL modalities at once
        batch_omics_data = {}
        for modality, tensor in raw_omics_data_dict.items():
            # Use CPU indexing to avoid CUDA errors
            cpu_tensor = tensor.cpu()
            batch_tensor = cpu_tensor[batch_patient_indices].to(device)
            batch_omics_data[modality] = batch_tensor

        # Reset gradients for this batch
        optimizer.zero_grad()

        # --- Process all modalities together with gradient accumulation --- #

        # Create augmented views for contrastive learning
        batch_omics_view1 = augment_raw_omics(batch_omics_data, args.aug_feature_mask_rate)
        batch_omics_view2 = augment_raw_omics(batch_omics_data, args.aug_feature_mask_rate)

        # Augment graph edges
        edge_index_dict_view1 = augment_edges(edge_index_dict, args.aug_edge_drop_rate)
        edge_index_dict_view2 = augment_edges(edge_index_dict, args.aug_edge_drop_rate)

        # Forward pass for both views (only once)
        try:
            if args.use_mixed_precision and device.type == 'cuda':
                # Note: Using older API for compatibility with remote cloud machine
                # The warning about using torch.amp.autocast('cuda') instead can be ignored
                with torch.cuda.amp.autocast():
                    final_embeddings_view1 = model(batch_omics_view1, gene_embeddings, edge_index_dict_view1)
                    final_embeddings_view2 = model(batch_omics_view2, gene_embeddings, edge_index_dict_view2)
            else:
                final_embeddings_view1 = model(batch_omics_view1, gene_embeddings, edge_index_dict_view1)
                final_embeddings_view2 = model(batch_omics_view2, gene_embeddings, edge_index_dict_view2)

            # --- Contrastive Loss (Patients) ---
            z_patient_view1 = final_embeddings_view1.get('patient')
            z_patient_view2 = final_embeddings_view2.get('patient')

            if z_patient_view1 is None or z_patient_view2 is None or z_patient_view1.shape[0] == 0:
                print(f"Warning: Patient embeddings missing or empty in output. Skipping contrastive loss.")
                loss_contrastive = torch.tensor(0.0, device=device)
                loss_risk_stratification = torch.tensor(0.0, device=device)
            else:
                loss_contrastive = contrastive_loss(z_patient_view1, z_patient_view2, temperature=args.contrastive_temp)

                # --- Risk Stratification Loss ---
                # Calculate risk stratification loss on view1 embeddings
                if args.risk_stratification_loss_weight > 0:
                    # Ensure we have enough samples for clustering (at least 2)
                    if z_patient_view1.shape[0] >= 2:
                        loss_risk_stratification = risk_stratification_loss(
                            z_patient_view1,
                            balance_weight=args.risk_balance_weight,
                            variance_weight=args.risk_variance_weight,
                            separation_weight=args.risk_separation_weight,
                            logrank_weight=args.risk_logrank_weight,
                            temperature=args.risk_temperature
                        )
                    else:
                        # Skip risk stratification loss if batch is too small
                        print(f"Warning: Batch size {z_patient_view1.shape[0]} too small for risk stratification. Skipping.")
                        loss_risk_stratification = torch.tensor(0.0, device=device)
                else:
                    loss_risk_stratification = torch.tensor(0.0, device=device)

            # --- Graph Reconstruction Loss (Genes - use view 1) ---
            loss_graph = torch.tensor(0.0, device=device)
            if args.graph_loss_weight > 0 and graph_loss_fn is not None:
                z_gene_view1 = final_embeddings_view1.get('gene')
                if z_gene_view1 is not None:
                    # Calculate with gradients if in reconstruction mode and gene embeddings are unfrozen
                    if args.training_strategy == 'reconstruction' and args.unfreeze_gene_embeddings:
                        adj_rec_logits = z_gene_view1 @ z_gene_view1.t()
                        loss_graph = graph_loss_fn(adj_rec_logits, graph_adj_tensor)
                    else:
                        # Original approach: Calculate without tracking gradients (for logging only)
                        with torch.no_grad():
                            adj_rec_logits = z_gene_view1 @ z_gene_view1.t()
                            loss_graph = graph_loss_fn(adj_rec_logits, graph_adj_tensor)

            # --- Structure Preservation Loss ---
            loss_structure = torch.tensor(0.0, device=device)
            loss_gene_interaction = torch.tensor(0.0, device=device)

            if args.unfreeze_gene_embeddings and 'gene' in final_embeddings_view1:
                z_gene_view1 = final_embeddings_view1['gene']

                # Calculate structure preservation loss
                if args.structure_loss_weight > 0:
                    loss_structure = structure_preservation_loss(z_gene_view1, original_gene_embeddings)

                # Calculate gene interaction loss
                if args.gene_interaction_loss_weight > 0 and graph_adj_tensor is not None:
                    loss_gene_interaction = gene_interaction_loss(
                        z_gene_view1,
                        graph_adj_tensor,
                        positive_weight=args.gene_interaction_pos_weight,
                        negative_weight=args.gene_interaction_neg_weight
                    )
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            continue

        # --- Process all modalities together for reconstruction loss --- #
        # Instead of cycling through modalities one by one, we'll process them all at once

        # Track modality-specific losses for logging
        modality_losses = []

        # Skip problematic modalities
        problematic_modalities = []
        if hasattr(model, 'problematic_modalities'):
            for modality, error_count in model.problematic_modalities.items():
                if error_count >= 3:  # Skip after 3 failures
                    print(f"Skipping problematic modality {modality} (failed {error_count} times)")
                    problematic_modalities.append(modality)
                    # Add a zero loss for this modality to maintain the count
                    modality_losses.append(torch.tensor(0.0, device=device))

        # Create a filtered batch_omics_data without problematic modalities
        filtered_batch_omics_data = {k: v for k, v in batch_omics_data.items() if k not in problematic_modalities}

        # Get patient and gene embeddings
        z_p = final_embeddings_view1.get('patient')
        z_gene = final_embeddings_view1.get('gene')

        if z_p is None or z_gene is None:
            print(f"Warning: Missing patient or gene embeddings. Skipping reconstruction.")
            loss_omics = torch.tensor(0.0, device=device)
        else:
            # Process all modalities together with a single decoder call
            try:
                if args.use_mixed_precision and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        # Call decode_omics without specific_modality to process all modalities
                        reconstructed_omics = model.decode_omics(final_embeddings_view1)
                else:
                    # Call decode_omics without specific_modality to process all modalities
                    reconstructed_omics = model.decode_omics(final_embeddings_view1)

                # Calculate loss for all modalities
                if isinstance(reconstructed_omics, dict):
                    # Handle dictionary output (modality-specific decoders)
                    total_mod_loss = 0.0
                    mod_count = 0

                    for mod_name, mod_tensor in reconstructed_omics.items():
                        if mod_name == 'concatenated':
                            continue  # Skip the concatenated key

                        if mod_name in filtered_batch_omics_data:
                            target_mod = filtered_batch_omics_data[mod_name]

                            # Apply gene masks if available
                            if gene_masks and not args.ignore_gene_masks and mod_name in gene_masks:
                                # Apply modality-specific mask
                                mask = torch.tensor(gene_masks[mod_name],
                                                  dtype=mod_tensor.dtype,
                                                  device=mod_tensor.device)
                                # Reshape mask for broadcasting
                                mask = mask.view(1, -1, 1)

                                # Ensure target_mod has the right shape for comparison
                                if mod_tensor.dim() == 3 and target_mod.dim() == 2:
                                    # Reshape target to [batch, genes, 1]
                                    target_mod = target_mod.unsqueeze(-1)

                                # Calculate masked MSE
                                if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                    # Both have final dimension 1, can compare directly
                                    squared_diff = (mod_tensor - target_mod) ** 2
                                else:
                                    # Need to average across genes first
                                    squared_diff = (mod_tensor.mean(dim=1) - target_mod.mean(dim=1, keepdim=True)) ** 2

                                # Apply mask
                                if squared_diff.dim() == mask.dim():
                                    masked_squared_diff = squared_diff * mask
                                else:
                                    # Adjust mask dimension if needed
                                    masked_squared_diff = squared_diff * mask.mean(dim=1, keepdim=True)

                                mask_sum = mask.sum()
                                if mask_sum > 0:
                                    mod_loss = masked_squared_diff.sum() / mask_sum
                                else:
                                    # Fallback with proper shape handling
                                    if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                        mod_loss = F.mse_loss(mod_tensor, target_mod)
                                    else:
                                        mod_loss = F.mse_loss(mod_tensor.mean(dim=1), target_mod.mean(dim=1))
                            else:
                                # Standard MSE for this modality with proper shape handling
                                if mod_tensor.dim() == 3 and target_mod.dim() == 2:
                                    # Reshape target to [batch, genes, 1]
                                    target_mod = target_mod.unsqueeze(-1)

                                if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                    # Both have final dimension 1, can compare directly
                                    mod_loss = F.mse_loss(mod_tensor, target_mod)
                                else:
                                    # Need to average across genes first
                                    mod_loss = F.mse_loss(mod_tensor.mean(dim=1), target_mod.mean(dim=1))

                            # Add to total loss
                            total_mod_loss += mod_loss.item()
                            mod_count += 1

                            # Store this modality's loss
                            modality_losses.append(mod_loss)

                    # Average loss across modalities
                    if mod_count > 0:
                        loss_omics = sum(modality_losses) / mod_count
                    else:
                        loss_omics = torch.tensor(0.0, device=device)
                else:
                    # Handle tensor output (concatenated decoder)
                    # This is the original approach with a single tensor output
                    loss_omics = F.mse_loss(reconstructed_omics, torch.cat([filtered_batch_omics_data[mod] for mod in modality_order], dim=1))

            except Exception as e:
                print(f"Error in joint decoding: {e}")
                import traceback
                traceback.print_exc()

                # Fallback to individual modality processing if joint decoding fails
                try:
                    reconstructed_omics = {}
                    modality_losses = []

                    for mod in filtered_batch_omics_data.keys():
                        try:
                            if args.use_mixed_precision and device.type == 'cuda':
                                with torch.cuda.amp.autocast():
                                    mod_result = model.decode_omics(final_embeddings_view1, specific_modality=mod)
                            else:
                                mod_result = model.decode_omics(final_embeddings_view1, specific_modality=mod)

                            if isinstance(mod_result, dict) and mod in mod_result:
                                mod_tensor = mod_result[mod]
                                target_mod = filtered_batch_omics_data[mod]

                                # Apply gene masks if available
                                if gene_masks and not args.ignore_gene_masks and mod in gene_masks:
                                    # Apply modality-specific mask
                                    mask = torch.tensor(gene_masks[mod],
                                                      dtype=mod_tensor.dtype,
                                                      device=mod_tensor.device)
                                    # Reshape mask for broadcasting
                                    mask = mask.view(1, -1, 1)

                                    # Ensure target_mod has the right shape for comparison
                                    if mod_tensor.dim() == 3 and target_mod.dim() == 2:
                                        # Reshape target to [batch, genes, 1]
                                        target_mod = target_mod.unsqueeze(-1)

                                    # Calculate masked MSE
                                    if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                        # Both have final dimension 1, can compare directly
                                        squared_diff = (mod_tensor - target_mod) ** 2
                                    else:
                                        # Need to average across genes first
                                        squared_diff = (mod_tensor.mean(dim=1) - target_mod.mean(dim=1, keepdim=True)) ** 2

                                    # Apply mask
                                    if squared_diff.dim() == mask.dim():
                                        masked_squared_diff = squared_diff * mask
                                    else:
                                        # Adjust mask dimension if needed
                                        masked_squared_diff = squared_diff * mask.mean(dim=1, keepdim=True)

                                    mask_sum = mask.sum()
                                    if mask_sum > 0:
                                        mod_loss = masked_squared_diff.sum() / mask_sum
                                    else:
                                        # Fallback with proper shape handling
                                        if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                            mod_loss = F.mse_loss(mod_tensor, target_mod)
                                        else:
                                            mod_loss = F.mse_loss(mod_tensor.mean(dim=1), target_mod.mean(dim=1))
                                else:
                                    # Standard MSE for this modality with proper shape handling
                                    if mod_tensor.dim() == 3 and target_mod.dim() == 2:
                                        # Reshape target to [batch, genes, 1]
                                        target_mod = target_mod.unsqueeze(-1)

                                    if mod_tensor.shape[-1] == 1 and target_mod.shape[-1] == 1:
                                        # Both have final dimension 1, can compare directly
                                        mod_loss = F.mse_loss(mod_tensor, target_mod)
                                    else:
                                        # Need to average across genes first
                                        mod_loss = F.mse_loss(mod_tensor.mean(dim=1), target_mod.mean(dim=1))

                                # Store this modality's loss
                                modality_losses.append(mod_loss)
                                reconstructed_omics[mod] = mod_tensor
                        except Exception as mod_e:
                            print(f"Error decoding modality {mod}: {mod_e}")
                            # Add a zero loss for this modality
                            modality_losses.append(torch.tensor(0.0, device=device))

                    # Average loss across modalities
                    if modality_losses:
                        loss_omics = sum(modality_losses) / len(modality_losses)
                    else:
                        loss_omics = torch.tensor(0.0, device=device)

                except Exception as fallback_e:
                    print(f"Error in fallback decoding: {fallback_e}")
                    loss_omics = torch.tensor(0.0, device=device)

                    # We don't need dimension adapters in the joint modality approach
                    # This code is no longer needed

                    # Apply dimension adapter if needed
                    if hasattr(model, 'dimension_adapters') and modality in model.dimension_adapters:
                        # Get embeddings
                        z_p_original = final_embeddings_view1.get('patient')
                        z_gene = final_embeddings_view1.get('gene')

                        # Apply adapter to patient embeddings
                        z_p_adapted = model.dimension_adapters[modality](z_p_original)

                        # Create new embeddings dict with adapted patient embeddings
                        adapted_embeddings = {
                            'patient': z_p_adapted,
                            'gene': z_gene
                        }

                        # Use adapted embeddings for this modality
                        if args.use_mixed_precision and device.type == 'cuda':
                            with torch.cuda.amp.autocast():
                                try:
                                    reconstructed_omics = model.decode_omics(adapted_embeddings, specific_modality=modality)
                                except Exception as e:
                                    print(f"Error with adapter: {e}")
                                    # Try direct decoding with omics_decoder as fallback
                                    # Create a completely new decoder for this modality
                                    if not hasattr(model, 'universal_decoder'):
                                        model.universal_decoder = True
                                        # Comment out these prints to reduce output clutter
                                        # print("Creating universal decoder for all modalities")

                                        # Create a fixed dimension adapter
                                        input_dim = z_p_adapted.shape[1]
                                        hidden_dim = 64

                                        # Create a universal patient decoder
                                        model.universal_patient_decoder = torch.nn.Sequential(
                                            torch.nn.Linear(input_dim, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim)
                                        ).to(z_p_adapted.device)

                                        # Create a universal reconstruction MLP
                                        combined_dim = hidden_dim + z_gene.shape[1]
                                        model.universal_reconstruction_mlp = torch.nn.Sequential(
                                            torch.nn.Linear(combined_dim, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1)
                                        ).to(z_p_adapted.device)

                                        # print(f"Created universal decoder with dimensions: input={input_dim}, hidden={hidden_dim}")

                                    # Use the universal decoder
                                    try:
                                        # 1. Get patient context
                                        patient_context = model.universal_patient_decoder(z_p_adapted)

                                        # 2. Process genes in chunks
                                        batch_size = z_p_adapted.shape[0]
                                        num_genes = z_gene.shape[0]
                                        genes_per_chunk = 10

                                        # Initialize result tensor
                                        result = torch.zeros(batch_size, num_genes, 1, device=z_p_adapted.device)

                                        # Process genes in chunks
                                        for g_start in range(0, num_genes, genes_per_chunk):
                                            g_end = min(g_start + genes_per_chunk, num_genes)
                                            genes_in_chunk = g_end - g_start

                                            # Get gene chunk
                                            z_gene_chunk = z_gene[g_start:g_end]

                                            # Expand dimensions for broadcasting
                                            patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)
                                            z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size, -1, -1)

                                            # Combine patient and gene embeddings
                                            combined = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                                            # Reshape for linear layer
                                            combined_flat = combined.reshape(-1, combined.size(-1))

                                            # Apply reconstruction MLP
                                            output_flat = model.universal_reconstruction_mlp(combined_flat)

                                            # Reshape back to 3D
                                            output = output_flat.reshape(batch_size, genes_in_chunk, 1)

                                            # Store in result tensor
                                            result[:, g_start:g_end, :] = output

                                        # Return result
                                        reconstructed_omics = {modality: result}

                                    except Exception as e:
                                        print(f"Error in universal decoder: {e}")
                                        # Fallback: return zeros
                                        reconstructed_omics = {
                                            modality: torch.zeros(z_p_adapted.shape[0], z_gene.shape[0], 1, device=z_p_adapted.device)
                                        }
                        else:
                            try:
                                reconstructed_omics = model.decode_omics(adapted_embeddings, specific_modality=modality)
                            except Exception as e:
                                print(f"Error with adapter: {e}")
                                # Try direct decoding with omics_decoder as fallback
                                # Create a completely new decoder for this modality
                                if not hasattr(model, 'universal_decoder'):
                                    model.universal_decoder = True
                                    # Comment out these prints to reduce output clutter
                                    # print("Creating universal decoder for all modalities")

                                    # Create a fixed dimension adapter
                                    input_dim = z_p_adapted.shape[1]
                                    hidden_dim = 64

                                    # Create a universal patient decoder
                                    model.universal_patient_decoder = torch.nn.Sequential(
                                        torch.nn.Linear(input_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim)
                                    ).to(z_p_adapted.device)

                                    # Create a universal reconstruction MLP
                                    combined_dim = hidden_dim + z_gene.shape[1]
                                    model.universal_reconstruction_mlp = torch.nn.Sequential(
                                        torch.nn.Linear(combined_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, 1)
                                    ).to(z_p_adapted.device)

                                    # print(f"Created universal decoder with dimensions: input={input_dim}, hidden={hidden_dim}")

                                # Use the universal decoder
                                try:
                                    # 1. Get patient context
                                    patient_context = model.universal_patient_decoder(z_p_adapted)

                                    # 2. Process genes in chunks
                                    batch_size = z_p_adapted.shape[0]
                                    num_genes = z_gene.shape[0]
                                    genes_per_chunk = 10

                                    # Initialize result tensor
                                    result = torch.zeros(batch_size, num_genes, 1, device=z_p_adapted.device)

                                    # Process genes in chunks
                                    for g_start in range(0, num_genes, genes_per_chunk):
                                        g_end = min(g_start + genes_per_chunk, num_genes)
                                        genes_in_chunk = g_end - g_start

                                        # Get gene chunk
                                        z_gene_chunk = z_gene[g_start:g_end]

                                        # Expand dimensions for broadcasting
                                        patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)
                                        z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size, -1, -1)

                                        # Combine patient and gene embeddings
                                        combined = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                                        # Reshape for linear layer
                                        combined_flat = combined.reshape(-1, combined.size(-1))

                                        # Apply reconstruction MLP
                                        output_flat = model.universal_reconstruction_mlp(combined_flat)

                                        # Reshape back to 3D
                                        output = output_flat.reshape(batch_size, genes_in_chunk, 1)

                                        # Store in result tensor
                                        result[:, g_start:g_end, :] = output

                                    # Return result
                                    reconstructed_omics = {modality: result}

                                except Exception as e:
                                    print(f"Error in universal decoder: {e}")
                                    # Fallback: return zeros
                                    reconstructed_omics = {
                                        modality: torch.zeros(z_p_adapted.shape[0], z_gene.shape[0], 1, device=z_p_adapted.device)
                                    }
                    else:
                        # Standard processing for other modalities
                        if args.use_mixed_precision and device.type == 'cuda':
                            # Note: Using older API for compatibility with remote cloud machine
                            # The warning about using torch.amp.autocast('cuda') instead can be ignored
                            with torch.cuda.amp.autocast():
                                # Try with specific_modality parameter first
                                try:
                                    reconstructed_omics = model.decode_omics(final_embeddings_view1, specific_modality=modality)
                                except TypeError:
                                    # Fall back to original method if specific_modality not supported
                                    print(f"Model doesn't support specific_modality parameter. Trying decode_single_modality instead.")
                                    if hasattr(model, 'decode_single_modality'):
                                        reconstructed_omics = model.decode_single_modality(final_embeddings_view1, modality)
                                    elif hasattr(model.omics_decoder, 'decode_single_modality'):
                                        # Get patient embeddings
                                        z_p = final_embeddings_view1.get('patient')
                                        z_gene = final_embeddings_view1.get('gene')

                                        # Create a completely new decoder for this modality
                                        if not hasattr(model, 'universal_decoder'):
                                            model.universal_decoder = True
                                            # Comment out these prints to reduce output clutter
                                            # print("Creating universal decoder for all modalities")

                                            # Create a fixed dimension adapter
                                            input_dim = z_p.shape[1]
                                            hidden_dim = 64

                                            # Create a universal patient decoder
                                            model.universal_patient_decoder = torch.nn.Sequential(
                                                torch.nn.Linear(input_dim, hidden_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_dim, hidden_dim)
                                            ).to(z_p.device)

                                            # Create a universal reconstruction MLP
                                            combined_dim = hidden_dim + z_gene.shape[1]
                                            model.universal_reconstruction_mlp = torch.nn.Sequential(
                                                torch.nn.Linear(combined_dim, hidden_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_dim, 1)
                                            ).to(z_p.device)

                                            # print(f"Created universal decoder with dimensions: input={input_dim}, hidden={hidden_dim}")

                                        # Use the universal decoder
                                        try:
                                            # 1. Get patient context
                                            patient_context = model.universal_patient_decoder(z_p)

                                            # 2. Process genes in chunks
                                            batch_size = z_p.shape[0]
                                            num_genes = z_gene.shape[0]
                                            genes_per_chunk = 10

                                            # Initialize result tensor
                                            result = torch.zeros(batch_size, num_genes, 1, device=z_p.device)

                                            # Process genes in chunks
                                            for g_start in range(0, num_genes, genes_per_chunk):
                                                g_end = min(g_start + genes_per_chunk, num_genes)
                                                genes_in_chunk = g_end - g_start

                                                # Get gene chunk
                                                z_gene_chunk = z_gene[g_start:g_end]

                                                # Expand dimensions for broadcasting
                                                patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)
                                                z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size, -1, -1)

                                                # Combine patient and gene embeddings
                                                combined = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                                                # Reshape for linear layer
                                                combined_flat = combined.reshape(-1, combined.size(-1))

                                                # Apply reconstruction MLP
                                                output_flat = model.universal_reconstruction_mlp(combined_flat)

                                                # Reshape back to 3D
                                                output = output_flat.reshape(batch_size, genes_in_chunk, 1)

                                                # Store in result tensor
                                                result[:, g_start:g_end, :] = output

                                            # Return result
                                            reconstructed_omics = result

                                        except Exception as e:
                                            print(f"Error in universal decoder: {e}")
                                            # Fallback: return zeros
                                            reconstructed_omics = torch.zeros(z_p.shape[0], z_gene.shape[0], 1, device=z_p.device)
                        else:
                            # Try with specific_modality parameter first
                            try:
                                reconstructed_omics = model.decode_omics(final_embeddings_view1, specific_modality=modality)
                            except TypeError:
                                # Fall back to original method if specific_modality not supported
                                print(f"Model doesn't support specific_modality parameter. Trying decode_single_modality instead.")
                                if hasattr(model, 'decode_single_modality'):
                                    reconstructed_omics = model.decode_single_modality(final_embeddings_view1, modality)
                                elif hasattr(model.omics_decoder, 'decode_single_modality'):
                                    # Get patient embeddings
                                    z_p = final_embeddings_view1.get('patient')
                                    z_gene = final_embeddings_view1.get('gene')

                                    # Create a completely new decoder for this modality
                                    if not hasattr(model, 'universal_decoder'):
                                        model.universal_decoder = True
                                        # Comment out these prints to reduce output clutter
                                        # print("Creating universal decoder for all modalities")

                                        # Create a fixed dimension adapter
                                        input_dim = z_p.shape[1]
                                        hidden_dim = 64

                                        # Create a universal patient decoder
                                        model.universal_patient_decoder = torch.nn.Sequential(
                                            torch.nn.Linear(input_dim, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim)
                                        ).to(z_p.device)

                                        # Create a universal reconstruction MLP
                                        combined_dim = hidden_dim + z_gene.shape[1]
                                        model.universal_reconstruction_mlp = torch.nn.Sequential(
                                            torch.nn.Linear(combined_dim, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1)
                                        ).to(z_p.device)

                                        # print(f"Created universal decoder with dimensions: input={input_dim}, hidden={hidden_dim}")

                                    # Use the universal decoder
                                    try:
                                        # 1. Get patient context
                                        patient_context = model.universal_patient_decoder(z_p)

                                        # 2. Process genes in chunks
                                        batch_size = z_p.shape[0]
                                        num_genes = z_gene.shape[0]
                                        genes_per_chunk = 10

                                        # Initialize result tensor
                                        result = torch.zeros(batch_size, num_genes, 1, device=z_p.device)

                                        # Process genes in chunks
                                        for g_start in range(0, num_genes, genes_per_chunk):
                                            g_end = min(g_start + genes_per_chunk, num_genes)
                                            genes_in_chunk = g_end - g_start

                                            # Get gene chunk
                                            z_gene_chunk = z_gene[g_start:g_end]

                                            # Expand dimensions for broadcasting
                                            patient_context_expanded = patient_context.unsqueeze(1).expand(-1, genes_in_chunk, -1)
                                            z_gene_expanded = z_gene_chunk.unsqueeze(0).expand(batch_size, -1, -1)

                                            # Combine patient and gene embeddings
                                            combined = torch.cat([patient_context_expanded, z_gene_expanded], dim=-1)

                                            # Reshape for linear layer
                                            combined_flat = combined.reshape(-1, combined.size(-1))

                                            # Apply reconstruction MLP
                                            output_flat = model.universal_reconstruction_mlp(combined_flat)

                                            # Reshape back to 3D
                                            output = output_flat.reshape(batch_size, genes_in_chunk, 1)

                                            # Store in result tensor
                                            result[:, g_start:g_end, :] = output

                                        # Return result
                                        reconstructed_omics = result

                                    except Exception as e:
                                        print(f"Error in universal decoder: {e}")
                                        # Fallback: return zeros
                                        reconstructed_omics = torch.zeros(z_p.shape[0], z_gene.shape[0], 1, device=z_p.device)
                except AttributeError as e:
                    print(f"AttributeError: {e}")
                    # If decode_single_modality is not available, we need to use a different approach
                    print(f"Warning: Model doesn't support modality-specific decoding. Using standard decode_omics.")
                    # Set the current_training_step attribute to control which modality is processed
                    if not hasattr(model, 'current_training_step'):
                        model.current_training_step = 0
                    # Find the index of the current modality in the modality order
                    if hasattr(model.omics_decoder, 'modality_order'):
                        modality_idx = model.omics_decoder.modality_order.index(modality)
                        model.current_training_step = modality_idx

                    # Now call decode_omics which should use the current_training_step to select the modality
                    if args.use_mixed_precision and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            reconstructed_omics = model.decode_omics(final_embeddings_view1)
                    else:
                        reconstructed_omics = model.decode_omics(final_embeddings_view1)

                # Get the reconstructed tensor for this modality
                if isinstance(reconstructed_omics, dict):
                    if modality in reconstructed_omics:
                        reconstructed_mod = reconstructed_omics[modality]
                    else:
                        # Use the first available modality
                        first_key = next(iter(reconstructed_omics))
                        reconstructed_mod = reconstructed_omics[first_key]
                        print(f"Warning: Modality {modality} not found in reconstructed output. Using {first_key} instead.")
                else:
                    reconstructed_mod = reconstructed_omics

                # Get target data for this modality
                target_mod = batch_omics_data[modality]

                # Calculate loss for this modality
                if gene_masks and not args.ignore_gene_masks and modality in gene_masks:
                    # Apply modality-specific mask
                    mask = torch.tensor(gene_masks[modality],
                                      dtype=reconstructed_mod.dtype,
                                      device=reconstructed_mod.device)
                    # Reshape mask for broadcasting
                    mask = mask.view(1, -1, 1)

                    # Ensure target_mod has the right shape for comparison
                    if reconstructed_mod.dim() == 3 and target_mod.dim() == 2:
                        # Reshape target to [batch, genes, 1]
                        target_mod = target_mod.unsqueeze(-1)

                    # Calculate masked MSE
                    if reconstructed_mod.shape[-1] == 1 and target_mod.shape[-1] == 1:
                        # Both have final dimension 1, can compare directly
                        squared_diff = (reconstructed_mod - target_mod) ** 2
                    else:
                        # Need to average across genes first
                        squared_diff = (reconstructed_mod.mean(dim=1) - target_mod.mean(dim=1, keepdim=True)) ** 2

                    # Apply mask
                    if squared_diff.dim() == mask.dim():
                        masked_squared_diff = squared_diff * mask
                    else:
                        # Adjust mask dimension if needed
                        masked_squared_diff = squared_diff * mask.mean(dim=1, keepdim=True)

                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        mod_loss = masked_squared_diff.sum() / mask_sum
                    else:
                        # Fallback with proper shape handling
                        if reconstructed_mod.shape[-1] == 1 and target_mod.shape[-1] == 1:
                            mod_loss = F.mse_loss(reconstructed_mod, target_mod)
                        else:
                            mod_loss = F.mse_loss(reconstructed_mod.mean(dim=1), target_mod.mean(dim=1))
                else:
                    # Standard MSE for this modality with proper shape handling
                    if reconstructed_mod.dim() == 3 and target_mod.dim() == 2:
                        # Reshape target to [batch, genes, 1]
                        target_mod = target_mod.unsqueeze(-1)

                    if reconstructed_mod.shape[-1] == 1 and target_mod.shape[-1] == 1:
                        # Both have final dimension 1, can compare directly
                        mod_loss = F.mse_loss(reconstructed_mod, target_mod)
                    else:
                        # Need to average across genes first
                        mod_loss = F.mse_loss(reconstructed_mod.mean(dim=1), target_mod.mean(dim=1))

                # Store this modality's loss
                modality_losses.append(mod_loss)

                # Clear memory if extreme memory efficiency is enabled
                if args.extreme_memory_efficient and device.type == 'cuda':
                    del reconstructed_omics, reconstructed_mod
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error processing modality {modality}: {e}")
                import traceback
                traceback.print_exc()

                # Track this modality as problematic
                if not hasattr(model, 'problematic_modalities'):
                    model.problematic_modalities = {}

                if modality not in model.problematic_modalities:
                    model.problematic_modalities[modality] = 1
                else:
                    model.problematic_modalities[modality] += 1

                print(f"Modality {modality} has failed {model.problematic_modalities[modality]} times")

                # Add a zero loss for this modality to maintain the count
                modality_losses.append(torch.tensor(0.0, device=device))

        # Average the modality losses
        if modality_losses:
            # Filter out any NaN losses
            valid_losses = [loss for loss in modality_losses if not torch.isnan(loss)]
            if valid_losses:
                loss_omics = sum(valid_losses) / len(valid_losses)
            else:
                loss_omics = torch.tensor(0.0, device=device)
        else:
            loss_omics = torch.tensor(0.0, device=device)

        # --- Combine Losses ---
        # Reconstruction-focused approach
        if args.training_strategy == 'reconstruction':
            # Focus on reconstruction losses
            batch_loss = (
                args.omics_loss_weight * loss_omics +
                args.graph_loss_weight * loss_graph
            )

            # Add structure preservation and gene interaction losses if using unfrozen embeddings
            if args.unfreeze_gene_embeddings:
                batch_loss += (
                    args.structure_loss_weight * loss_structure +
                    args.gene_interaction_loss_weight * loss_gene_interaction
                )

            # Add risk stratification loss with highest weight
            if args.risk_stratification_loss_weight > 0:
                # Ensure loss_risk_stratification is a scalar tensor
                if loss_risk_stratification.dim() > 0:
                    loss_risk_stratification = loss_risk_stratification.mean()
                batch_loss = batch_loss + args.risk_stratification_loss_weight * loss_risk_stratification

            # Optionally add small contrastive loss if desired
            if args.contrastive_loss_weight > 0:
                batch_loss += args.contrastive_loss_weight * loss_contrastive
        else:
            # Original contrastive approach with risk stratification
            batch_loss = (
                loss_contrastive +
                args.omics_loss_weight * loss_omics
            )

            # Add risk stratification loss with highest weight
            if args.risk_stratification_loss_weight > 0:
                # Ensure loss_risk_stratification is a scalar tensor
                if loss_risk_stratification.dim() > 0:
                    loss_risk_stratification = loss_risk_stratification.mean()
                batch_loss = batch_loss + args.risk_stratification_loss_weight * loss_risk_stratification

        # Check for NaN or Inf values
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping backward pass.")
            continue

        # Scale the loss by the number of gradient accumulation steps
        scaled_batch_loss = batch_loss / grad_accum_steps

        # Perform backward pass with gradient accumulation
        if args.use_mixed_precision and device.type == 'cuda' and scaler is not None:
            # Scale the loss and accumulate gradients
            scaler.scale(scaled_batch_loss).backward()

            # Only update weights after accumulating gradients for grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                # Apply gradient clipping
                if args.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)

                # Update weights and reset gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Backward pass with gradient accumulation
            scaled_batch_loss.backward()

            # Only update weights after accumulating gradients for grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                # Apply gradient clipping
                if args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)

                # Update weights and reset gradients
                optimizer.step()
                optimizer.zero_grad()

        # Accumulate losses for reporting
        total_loss += batch_loss.item()
        contrastive_loss_sum += loss_contrastive.item()
        graph_loss_sum += loss_graph.item()
        omics_loss_sum += loss_omics.item()
        structure_loss_sum += loss_structure.item()
        gene_interaction_loss_sum += loss_gene_interaction.item()
        risk_stratification_loss_sum += loss_risk_stratification.item()

        # Update progress bar
        batch_pbar.set_postfix({
            'loss': batch_loss.item(),
            'cont': loss_contrastive.item(),
            'graph': loss_graph.item(),
            'omics': loss_omics.item(),
            'struct': loss_structure.item(),
            'gene_int': loss_gene_interaction.item(),
            'risk_strat': loss_risk_stratification.item()
        })
        batch_pbar.update(1)

        # Clear memory if extreme memory efficiency is enabled
        if args.extreme_memory_efficient and device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    # Close batch progress bar
    batch_pbar.close()

    # Calculate average losses
    avg_loss = total_loss / max(1, num_batches)
    avg_contrastive_loss = contrastive_loss_sum / max(1, num_batches)
    avg_graph_loss = graph_loss_sum / max(1, num_batches)
    avg_omics_loss = omics_loss_sum / max(1, num_batches)
    avg_structure_loss = structure_loss_sum / max(1, num_batches)
    avg_gene_interaction_loss = gene_interaction_loss_sum / max(1, num_batches)
    avg_risk_stratification_loss = risk_stratification_loss_sum / max(1, num_batches)

    # Return metrics
    return {
        'loss': avg_loss,
        'contrastive_loss': avg_contrastive_loss,
        'graph_loss': avg_graph_loss,
        'omics_loss': avg_omics_loss,
        'structure_loss': avg_structure_loss,
        'gene_interaction_loss': avg_gene_interaction_loss,
        'risk_stratification_loss': avg_risk_stratification_loss
    }

def run_integrated_training_improved(args):
    """
    Main function to run the improved training with modality cycling.

    Args:
        args: Command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()

    # --- W&B Init --- #
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Include training strategy in run name
    strategy_tag = "recon" if args.training_strategy == 'reconstruction' else "contrast"
    gene_emb_tag = "unfrozen" if args.unfreeze_gene_embeddings else "frozen"

    run_name = f"integ-improved-{strategy_tag}-{gene_emb_tag}-{args.cancer_type}-{run_timestamp}"
    project_name = f"integrated-trans-gcn-{args.cancer_type}"

    print("Initializing Weights & Biases...")
    try:
        # Add custom tags for easier filtering
        tags = [
            "improved-training",
            args.training_strategy,
            f"gene_emb_{gene_emb_tag}",
            f"gcn_{args.gcn_conv_type}",
            f"T{args.transformer_layers}L_G{args.gcn_layers}L"
        ]

        wandb.init(project=project_name, name=run_name, config=vars(args), tags=tags)
        print(f"W&B Run URL: {wandb.run.get_url()}")
    except Exception as e:
        print(f"Error initializing W&B: {e}. Proceeding without W&B tracking.")
        os.environ["WANDB_DISABLED"] = "true"
        wandb.init(mode="disabled")

    # --- Setup AMP for mixed precision training if enabled ---
    if args.use_mixed_precision and device.type == 'cuda':
        print("\nUsing Automatic Mixed Precision (AMP) for training...")
        # Use the older torch.cuda.amp.GradScaler API for compatibility with remote cloud machine
        # The warning about using torch.amp.GradScaler() instead can be ignored
        scaler = torch.cuda.amp.GradScaler()
        # Set PyTorch to allocate memory more conservatively
        torch.cuda.empty_cache()
        gc.collect()
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = args.cuda_alloc_conf
    else:
        scaler = None

    # For any CUDA device, set the memory allocation configuration
    if device.type == 'cuda' and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = args.cuda_alloc_conf
        print(f"Setting PYTORCH_CUDA_ALLOC_CONF={args.cuda_alloc_conf}")
        torch.cuda.empty_cache()  # Clear any existing allocations
        gc.collect()

    # --- Load Data --- #
    # Load Raw Omics
    raw_omics_data_dict, patient_ids = load_raw_omics_data(args.original_data_path, args.cancer_type)
    if raw_omics_data_dict is None: return

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
    gene_embeddings, gene_emb_list = load_gene_embeddings(args.gene_embedding_path)
    if gene_embeddings is None: return
    gene_embeddings = gene_embeddings.to(device)
    print(f"Moved gene embeddings to {device}")

    # Store original gene embeddings for structure preservation loss
    original_gene_embeddings = gene_embeddings.clone().detach()

    # Allow gene embeddings to be updated with controlled learning
    if args.unfreeze_gene_embeddings:
        gene_embeddings.requires_grad_(True)
        print("STRATEGY 4: Using controlled updates to gene embeddings with structure preservation.")
    else:
        # Keep frozen as in original implementation
        gene_embeddings.requires_grad_(False)
        print("STRATEGY 3: Frozen pre-trained gene embeddings. They will not be updated during GCN training.")

    # Load Gene Interactions and Masks
    adj_matrix_gg, interaction_gene_list, gene_masks = load_gene_interactions_and_masks(
        args.original_data_path, args.cancer_type
    )
    if adj_matrix_gg is None: return

    # --- Gene List Consistency Check --- #
    if gene_emb_list != interaction_gene_list:
        print("\nCritical Warning: Gene lists from gene embeddings and gene interactions do NOT match!")
        # Attempt to align if possible, otherwise error or proceed with caution
        common_genes = sorted(list(set(gene_emb_list) & set(interaction_gene_list)))
        if not common_genes:
             print("Error: No common genes between embedding list and interaction list. Cannot proceed.")
             return
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
             # Proceeding might lead to errors or incorrect graph structure.

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

    # --- Construct Graph Edge Index Dictionary --- #
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
    if args.pg_link_omics and args.pg_link_omics in raw_omics_data_dict:
         # Need to convert the tensor back to DataFrame for the function
         # Or adapt create_patient_gene_edges to work with tensors
         print(f"Using raw {args.pg_link_omics} tensor for patient-gene links...")
         omics_tensor_pg = raw_omics_data_dict[args.pg_link_omics].cpu() # Move to CPU for pandas
         # Create DataFrame with correct index/columns
         omics_df_for_links = pd.DataFrame(omics_tensor_pg.numpy(), index=patient_ids, columns=gene_list[:omics_tensor_pg.shape[1]]) # Assumes gene order matches columns

         # Check if gene list used for columns matches the full gene_list
         if omics_df_for_links.shape[1] != len(gene_list):
              print(f"Warning: Number of features in {args.pg_link_omics} ({omics_df_for_links.shape[1]}) does not match final gene list ({len(gene_list)}). Cannot create P-G links accurately this way.")
              omics_df_for_links = None # Prevent using mismatched data

    elif args.pg_link_omics:
         print(f"Warning: Specified pg_link_omics '{args.pg_link_omics}' not found in loaded raw data.")

    if omics_df_for_links is not None:
        edge_index_pg = create_patient_gene_edges(omics_df_for_links, patient_ids, gene_list,
                                                  link_type=args.pg_link_type,
                                                  threshold=args.pg_link_threshold,
                                                  top_k=args.pg_link_top_k)
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

    # --- Prepare Targets for Reconstruction Losses ---
    # Prepare binary graph adjacency target tensor
    if adj_matrix_gg is not None:
        if sp.issparse(adj_matrix_gg):
            adj_matrix_binary_target = adj_matrix_gg.astype(bool).astype(float)
        else: # Handle numpy array case
            adj_matrix_binary_target = (adj_matrix_gg > 0).astype(float)
        graph_adj_tensor = torch.tensor(adj_matrix_binary_target.todense() if sp.issparse(adj_matrix_binary_target) else adj_matrix_binary_target, dtype=torch.float32).to(device)
    else:
        graph_adj_tensor = None

    # --- Model Instantiation --- #
    print("\nInstantiating IntegratedTransformerGCN model...")

    # Load modality latent dimensions from JSON file if provided
    modality_latent_dims = None
    if args.modality_latents_path:
        try:
            with open(args.modality_latents_path, 'r') as f:
                modality_latent_dims = json.load(f)
            # Basic validation: check if it's a dict and values are int
            if not isinstance(modality_latent_dims, dict) or not all(isinstance(v, int) for v in modality_latent_dims.values()):
                raise ValueError("JSON file must contain a dictionary mapping modality names to integer dimensions.")

            # Ensure all modality names are lowercase for case-insensitive matching
            modality_latent_dims = {k.lower(): v for k, v in modality_latent_dims.items()}

            print(f"\nModality names in config (after lowercase): {list(modality_latent_dims.keys())}")
            print(f"Modality names in data (lowercase): {[k.lower() for k in omics_input_dims.keys()]}")

            # Verify that all modalities in the data are in the config
            missing_modalities = set(k.lower() for k in omics_input_dims.keys()) - set(modality_latent_dims.keys())
            if missing_modalities:
                print(f"Warning: The following modalities are missing from the config: {missing_modalities}")
                print("These modalities will use automatic dimension calculation.")

            # Verify that all dimensions are positive
            zero_dims = [k for k, v in modality_latent_dims.items() if v <= 0]
            if zero_dims:
                print(f"Warning: The following modalities have zero or negative dimensions in the config: {zero_dims}")
                print("Setting these dimensions to a minimum value of 16.")
                for k in zero_dims:
                    modality_latent_dims[k] = 16

            print(f"Using modality latent dimensions from {args.modality_latents_path}: {modality_latent_dims}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing modality_latents JSON file: {e}")
            print("Falling back to automatic latent dimension calculation")
            modality_latent_dims = None
        except FileNotFoundError:
            print(f"Error: Could not find the modality_latents JSON file at {args.modality_latents_path}")
            print("Falling back to automatic latent dimension calculation")
            modality_latent_dims = None

    # Determine if omics decoder should be added based on loss weight
    add_decoder = args.omics_loss_weight > 0

    model = IntegratedTransformerGCN(
        # Transformer Args
        omics_input_dims=omics_input_dims,
        transformer_embed_dim=args.transformer_embed_dim,
        transformer_num_heads=args.transformer_num_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        num_transformer_layers=args.transformer_layers,
        transformer_output_dim=args.transformer_output_dim, # This acts as GCN patient input dim
        transformer_dropout=args.transformer_dropout,
        # GCN Args
        gcn_metadata=metadata,
        gene_feature_dim=gene_embeddings.shape[1], # Add gene feature dimension from embeddings
        gcn_hidden_channels=args.gcn_hidden_dim, # Renamed arg
        gcn_out_channels=args.gcn_output_dim, # Renamed arg
        gcn_num_layers=args.gcn_layers,
        gcn_conv_type=args.gcn_conv_type,
        gcn_num_heads=args.gcn_gat_heads,
        gcn_dropout_rate=args.gcn_dropout,
        gcn_use_layer_norm=not args.gcn_no_norm,
        gene_masks=gene_masks if not args.ignore_gene_masks else None, # Pass masks
        # Decoder Args
        add_omics_decoder=add_decoder,
        use_modality_specific_decoders=args.use_modality_specific_decoders,
        decoder_activation=args.decoder_activation,
        decoder_patient_batch_size=args.decoder_patient_batch_size,
        num_genes=num_genes, # Pass num_genes explicitly
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        reduce_decoder_memory=args.reduce_decoder_memory,
        decoder_mlp_factor=args.decoder_mlp_factor, # Add new parameter to control decoder MLP size
        genes_per_chunk=args.genes_per_chunk, # Add parameter to control chunk size
        use_mixed_precision=args.use_mixed_precision,
        extreme_memory_efficient=args.extreme_memory_efficient, # Enable extreme memory efficiency
        modality_latent_dims=modality_latent_dims, # Custom modality latent dimensions from config file
        modality_by_modality=args.modality_by_modality # Process one modality at a time
    ).to(device)

    # Store modality order for later use in loss calculation
    if add_decoder:
        args.decoder_modality_order = sorted(omics_input_dims.keys())

    # Enable modality-by-modality training if requested
    if args.modality_by_modality:
        print("\nUsing modality-by-modality training approach to reduce memory usage")
        print(f"Will cycle through {len(omics_input_dims)} modalities: {list(omics_input_dims.keys())}")
        # Force use_modality_specific_decoders to True
        args.use_modality_specific_decoders = True
        print("Enabled modality-specific decoders for modality-by-modality training")

    # --- Initialize Lazy Modules (if any) --- #
    # Perform a dummy forward pass
    print("Initializing lazy modules (if any)...")
    with torch.no_grad():
        try:
            # Method 1: Use a full forward pass with the complete dataset to initialize everything
            _ = model(raw_omics_data_dict, gene_embeddings, edge_index_dict)
            print("Lazy modules initialized with full dataset.")
        except Exception as e:
            print(f"Warning: Error initializing with full dataset: {e}")
            print("Initializing model parameters directly...")
            # Method 2: Force initialization of all LazyModules by ensuring every parameter is initialized
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

    # --- Training --- #
    if args.train_epochs <= 0:
        print("\nWarning: train_epochs set to 0. Skipping training.")
        return

    # Initialize training variables
    start_epoch = 0
    training_losses = []

    # Check if we're resuming from a checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        # Create parameter groups with different learning rates (needed for optimizer initialization)
        if args.unfreeze_gene_embeddings:
            # Separate gene embeddings for controlled learning rate
            optimizer = optim.Adam([
                {'params': [p for _, p in model.named_parameters()], 'lr': args.gcn_lr},
                {'params': [gene_embeddings], 'lr': args.gene_lr}
            ], weight_decay=args.gcn_weight_decay)
        else:
            # Standard optimizer (gene embeddings are frozen)
            optimizer = optim.Adam(model.parameters(), lr=args.gcn_lr, weight_decay=args.gcn_weight_decay)

        # Load checkpoint
        start_epoch, training_losses, loaded_gene_embeddings, _ = load_checkpoint(
            args.checkpoint_path, model, optimizer, device, scaler
        )

        # Restore gene embeddings if they were trainable and saved
        if args.unfreeze_gene_embeddings and loaded_gene_embeddings is not None:
            print("Restoring trainable gene embeddings from checkpoint")
            gene_embeddings = loaded_gene_embeddings.to(device)
            gene_embeddings.requires_grad_(True)

        # Log checkpoint resumption
        print(f"\n--- Resuming Training from Epoch {start_epoch} to {args.train_epochs} ---")
        if wandb.run and wandb.run.mode != "disabled":
            wandb.log({"Checkpoint/Resumed_From_Epoch": start_epoch - 1})
    else:
        if args.checkpoint_path:
            print(f"Warning: Checkpoint path {args.checkpoint_path} not found. Starting training from scratch.")

        print(f"\n--- Training Integrated Model for {args.train_epochs} epochs ---")

        # Create parameter groups with different learning rates
        if args.unfreeze_gene_embeddings:
            # Separate gene embeddings for controlled learning rate
            optimizer = optim.Adam([
                {'params': [p for _, p in model.named_parameters()], 'lr': args.gcn_lr},
                {'params': [gene_embeddings], 'lr': args.gene_lr}
            ], weight_decay=args.gcn_weight_decay)
            print(f"Using learning rate {args.gcn_lr} for model parameters and {args.gene_lr} for gene embeddings")
        else:
            # Standard optimizer (gene embeddings are frozen)
            optimizer = optim.Adam(model.parameters(), lr=args.gcn_lr, weight_decay=args.gcn_weight_decay)
            print(f"Using learning rate {args.gcn_lr} for model parameters (gene embeddings frozen)")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    train_pbar = tqdm(range(start_epoch, args.train_epochs), desc='Training Epochs')

    # Early stopping variables
    best_loss = float('inf')
    best_model_state = None
    best_gene_embeddings = None
    early_stopping_counter = 0
    early_stopping_triggered = False

    # Print early stopping status
    if args.early_stopping:
        print(f"\nEarly stopping enabled with patience={args.early_stopping_patience} and min_delta={args.early_stopping_min_delta}")
    else:
        print("\nEarly stopping is disabled")

    for epoch in train_pbar:
        # Train for one epoch with joint modality processing
        train_metrics = train_with_joint_modalities(
            model=model,
            optimizer=optimizer,
            raw_omics_data_dict=raw_omics_data_dict,
            gene_embeddings=gene_embeddings,
            edge_index_dict=edge_index_dict,
            graph_adj_tensor=graph_adj_tensor,
            gene_masks=gene_masks,
            original_gene_embeddings=original_gene_embeddings,
            args=args,
            device=device,
            scaler=scaler
        )

        # Update training losses
        current_loss = train_metrics['loss']
        training_losses.append(current_loss)

        # Update progress bar
        train_pbar.set_postfix({
            'loss': current_loss,
            'cont': train_metrics['contrastive_loss'],
            'graph': train_metrics['graph_loss'],
            'omics': train_metrics['omics_loss'],
            'struct': train_metrics['structure_loss'],
            'gene_int': train_metrics['gene_interaction_loss'],
            'risk': train_metrics['risk_stratification_loss']
        })

        # Log to W&B
        if wandb.run and wandb.run.mode != "disabled":
            wandb.log({
                "Epoch": epoch,
                "Loss/Total": current_loss,
                "Loss/Contrastive": train_metrics['contrastive_loss'],
                "Loss/Graph": train_metrics['graph_loss'],
                "Loss/Omics": train_metrics['omics_loss'],
                "Loss/Structure": train_metrics['structure_loss'],
                "Loss/Gene_Interaction": train_metrics['gene_interaction_loss'],
                "Loss/Risk_Stratification": train_metrics['risk_stratification_loss']
            })

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.train_epochs - 1:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                training_losses=training_losses,
                gene_embeddings=gene_embeddings,
                args=args,
                output_dir=args.output_dir,
                scaler=scaler
            )

        # Early stopping logic
        if args.early_stopping:
            # Check if current loss is better than best loss
            if current_loss < best_loss - args.early_stopping_min_delta:
                # We have an improvement
                best_loss = current_loss
                early_stopping_counter = 0

                # Save the best model state and gene embeddings
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if gene_embeddings.requires_grad:
                    best_gene_embeddings = gene_embeddings.detach().cpu().clone()
                else:
                    best_gene_embeddings = gene_embeddings.cpu().clone()

                # Save a special checkpoint for the best model
                best_model_path = os.path.join(args.output_dir, f"best_model_checkpoint.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gene_embeddings': gene_embeddings.detach().cpu() if gene_embeddings.requires_grad else None,
                    'best_loss': best_loss,
                    'args': vars(args)
                }, best_model_path)

                # Log the new best model
                print(f"\nEpoch {epoch}: New best model with loss {best_loss:.6f}")
                print(f"Best model saved to {best_model_path}")
                if wandb.run and wandb.run.mode != "disabled":
                    wandb.log({"Early_Stopping/Best_Loss": best_loss, "Early_Stopping/Best_Epoch": epoch})
            else:
                # No improvement
                early_stopping_counter += 1
                print(f"\nEpoch {epoch}: No improvement for {early_stopping_counter} epochs. Best loss: {best_loss:.6f}")
                if wandb.run and wandb.run.mode != "disabled":
                    wandb.log({"Early_Stopping/Counter": early_stopping_counter})

                # Check if we should stop training
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    if wandb.run and wandb.run.mode != "disabled":
                        wandb.log({"Early_Stopping/Triggered": True, "Early_Stopping/Stop_Epoch": epoch})

                    # Restore best model and gene embeddings
                    if best_model_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        if gene_embeddings.requires_grad and best_gene_embeddings is not None:
                            gene_embeddings.data = best_gene_embeddings.to(device)
                        print(f"Restored best model from epoch {epoch - early_stopping_counter}")

                    early_stopping_triggered = True
                    break

    # Save final model
    final_model_path = os.path.join(args.output_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    # If early stopping was triggered, add a note to the filename
    if args.early_stopping and early_stopping_triggered:
        final_model_path = os.path.join(args.output_dir, f"final_model_early_stopped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        print(f"\nSaving early-stopped model (best model from training)")
    else:
        print(f"\nSaving final model from last epoch")

    torch.save({
        'model_state_dict': model.state_dict(),
        'gene_embeddings': gene_embeddings.detach().cpu() if gene_embeddings.requires_grad else None,
        'args': vars(args),
        'early_stopped': args.early_stopping and early_stopping_triggered,
        'best_loss': best_loss if args.early_stopping else None,
        'stopped_epoch': epoch if args.early_stopping and early_stopping_triggered else None
    }, final_model_path)
    print(f"Model saved to {final_model_path}")

    # Generate and save final embeddings
    print("\n--- Generating and Saving Final Embeddings ---")
    model.eval()
    with torch.no_grad():
        try:
            print("Generating final embeddings using batched processing...")
            # Process in batches to avoid memory issues
            batch_size = args.patient_batch_size
            num_patients = len(patient_ids)
            num_batches = (num_patients + batch_size - 1) // batch_size

            # Initialize lists to collect embeddings
            all_patient_embeddings = []
            final_gene_embeddings = None

            # Process each batch
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_patients)
                batch_patient_indices = list(range(start_idx, end_idx))

                # Extract batch data
                batch_omics_data = {}
                for modality, tensor in raw_omics_data_dict.items():
                    # Use CPU indexing to avoid CUDA errors
                    cpu_tensor = tensor.cpu()
                    batch_tensor = cpu_tensor[batch_patient_indices].to(device)
                    batch_omics_data[modality] = batch_tensor

                # Forward pass for this batch
                batch_embeddings_dict = model(batch_omics_data, gene_embeddings, edge_index_dict)

                # Extract and store patient embeddings
                batch_patient_embeddings = batch_embeddings_dict.get('patient')
                if batch_patient_embeddings is not None:
                    all_patient_embeddings.append(batch_patient_embeddings.cpu().numpy())

                # Store gene embeddings (only need to do this once as they're shared)
                if final_gene_embeddings is None:
                    final_gene_embeddings = batch_embeddings_dict.get('gene')
                    if final_gene_embeddings is not None:
                        final_gene_embeddings = final_gene_embeddings.cpu().numpy()

                # Clear memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            # Combine patient embeddings from all batches
            if all_patient_embeddings:
                final_patient_embeddings = np.concatenate(all_patient_embeddings, axis=0)
                print(f"Final patient embeddings shape: {final_patient_embeddings.shape}")

                # Save results
                results_data = {
                    'patient_ids': patient_ids,
                    'final_patient_embeddings_gcn': final_patient_embeddings,
                    'gene_list': gene_list,  # Final aligned gene list
                    'gene_embeddings': final_gene_embeddings,
                    'training_losses': training_losses,
                    'args': vars(args),
                    'early_stopped': args.early_stopping and early_stopping_triggered,
                    'best_loss': best_loss if args.early_stopping else None,
                    'stopped_epoch': epoch if args.early_stopping and early_stopping_triggered else None
                }

                # Create a more descriptive filename if early stopping was triggered
                if args.early_stopping and early_stopping_triggered:
                    embeddings_save_path = os.path.join(args.output_dir, f'integrated_embeddings_{args.cancer_type}_early_stopped.joblib')
                else:
                    embeddings_save_path = os.path.join(args.output_dir, f'integrated_embeddings_{args.cancer_type}.joblib')

                joblib.dump(results_data, embeddings_save_path)
                print(f"Final embeddings saved to {embeddings_save_path}")

                # Log to wandb if embeddings were successfully saved
                if wandb.run and wandb.run.mode != "disabled":
                    emb_artifact = wandb.Artifact(
                        name=f"integrated-embeddings-{args.cancer_type}", type="embeddings",
                        description=f"Final patient/gene embeddings from IntegratedTransformerGCN for {args.cancer_type}"
                    )
                    emb_artifact.add_file(embeddings_save_path)
                    wandb.log_artifact(emb_artifact)
            else:
                print("Error: No patient embeddings were generated in any batch.")
        except Exception as e:
            print(f"Error generating final embeddings: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for better debugging

    # Log to W&B
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({"Training/Final_Loss": training_losses[-1]})

        # Log early stopping summary if enabled
        if args.early_stopping:
            wandb.run.summary["early_stopping_enabled"] = True
            wandb.run.summary["early_stopping_best_loss"] = best_loss
            wandb.run.summary["early_stopping_triggered"] = early_stopping_triggered
            if early_stopping_triggered:
                wandb.run.summary["early_stopping_stopped_epoch"] = epoch
                wandb.run.summary["early_stopping_best_epoch"] = epoch - early_stopping_counter

        wandb.finish()

    print(f"\nTraining completed in {(time.time() - start_time) / 60:.2f} minutes")
    return model, gene_embeddings

# --- Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Integrated Transformer-GCN Model with Improved Training")

    # Data arguments
    parser.add_argument("--original_data_path", type=str, required=True, help="Path to original data joblib file")
    parser.add_argument("--gene_embedding_path", type=str, required=True, help="Path to pre-trained gene embeddings")
    parser.add_argument("--cancer_type", type=str, required=True, help="Cancer type to use (e.g., 'colorec', 'panc')")
    parser.add_argument("--output_dir", type=str, default="results/integrated_transformer_gcn", help="Output directory for results")

    # Transformer arguments
    parser.add_argument("--transformer_embed_dim", type=int, default=128, help="Embedding dimension for transformer")
    parser.add_argument("--transformer_num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--transformer_ff_dim", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--transformer_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--transformer_output_dim", type=int, default=64, help="Output dimension from transformer")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for transformer")

    # GCN arguments
    parser.add_argument("--gcn_hidden_dim", type=int, default=256, help="Hidden dimension for GCN")
    parser.add_argument("--gcn_output_dim", type=int, default=128, help="Output dimension for GCN")
    parser.add_argument("--gcn_layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--gcn_conv_type", type=str, default="sage", choices=["gcn", "sage", "gat"], help="GCN convolution type")
    parser.add_argument("--gcn_gat_heads", type=int, default=4, help="Number of attention heads for GAT")
    parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout rate for GCN")
    parser.add_argument("--gcn_no_norm", action="store_true", help="Disable layer normalization in GCN")

    # Patient-gene edge arguments
    parser.add_argument("--pg_link_omics", type=str, default=None, help="Omics modality to use for patient-gene links")
    parser.add_argument("--pg_link_type", type=str, default="threshold",
                        choices=["threshold", "top_k_per_patient", "top_k_per_gene"],
                        help="Method to create patient-gene links")
    parser.add_argument("--pg_link_threshold", type=float, default=0.5, help="Threshold for patient-gene links")
    parser.add_argument("--pg_link_top_k", type=int, default=10, help="Top-k for patient-gene links")

    # Training arguments
    parser.add_argument("--train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patient_batch_size", type=int, default=32, help="Batch size for patients")
    parser.add_argument("--gcn_lr", type=float, default=0.001, help="Learning rate for GCN")
    parser.add_argument("--gcn_weight_decay", type=float, default=0.0001, help="Weight decay for GCN")
    parser.add_argument("--training_strategy", type=str, default="reconstruction", choices=["contrastive", "reconstruction"], help="Training strategy")
    parser.add_argument("--contrastive_temp", type=float, default=0.1, help="Temperature for contrastive loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.0, help="Weight for contrastive loss in reconstruction mode")
    parser.add_argument("--omics_loss_weight", type=float, default=1.0, help="Weight for omics reconstruction loss")
    parser.add_argument("--graph_loss_weight", type=float, default=0.1, help="Weight for graph reconstruction loss")
    parser.add_argument("--risk_logrank_weight", type=float, default=0.4, help="Weight for log-rank approximation term in risk stratification loss")
    parser.add_argument("--structure_loss_weight", type=float, default=0.1, help="Weight for structure preservation loss")
    parser.add_argument("--gene_interaction_loss_weight", type=float, default=0.1, help="Weight for gene interaction loss")
    parser.add_argument("--gene_interaction_pos_weight", type=float, default=1.0, help="Positive weight for gene interaction loss")
    parser.add_argument("--gene_interaction_neg_weight", type=float, default=0.1, help="Negative weight for gene interaction loss")
    parser.add_argument("--aug_feature_mask_rate", type=float, default=0.1, help="Feature masking rate for augmentation")
    parser.add_argument("--aug_edge_drop_rate", type=float, default=0.1, help="Edge dropping rate for augmentation")

    # Risk stratification loss arguments
    parser.add_argument("--risk_stratification_loss_weight", type=float, default=1.0, help="Weight for risk stratification loss (highest weight recommended)")
    parser.add_argument("--risk_balance_weight", type=float, default=0.3, help="Weight for cluster balance term in risk stratification loss")
    parser.add_argument("--risk_variance_weight", type=float, default=0.3, help="Weight for within-cluster variance term in risk stratification loss")
    parser.add_argument("--risk_separation_weight", type=float, default=0.4, help="Weight for between-cluster separation term in risk stratification loss")
    parser.add_argument("--risk_temperature", type=float, default=0.1, help="Temperature parameter for soft clustering in risk stratification loss")
    parser.add_argument("--unfreeze_gene_embeddings", action="store_true", help="Allow gene embeddings to be updated")
    parser.add_argument("--gene_lr", type=float, default=0.0001, help="Learning rate for gene embeddings")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_optimizer", action="store_true", help="Save optimizer state in checkpoint")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm")

    # Early stopping arguments
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001, help="Minimum change in loss to qualify as an improvement")

    # Decoder arguments
    parser.add_argument("--use_modality_specific_decoders", action="store_true", help="Use separate decoder for each modality")
    parser.add_argument("--decoder_activation", type=str, default="sigmoid", choices=["sigmoid", "relu", "none"], help="Activation function for decoder")
    parser.add_argument("--decoder_patient_batch_size", type=int, default=32, help="Batch size for decoder")
    parser.add_argument("--reduce_decoder_memory", action="store_true", help="Reduce memory usage in decoder")
    parser.add_argument("--decoder_mlp_factor", type=float, default=1.0, help="Factor to scale decoder MLP dimensions")
    parser.add_argument("--genes_per_chunk", type=int, default=10, help="Number of genes to process at once in decoder")
    parser.add_argument("--modality_latents_path", type=str, default=None, help="Path to JSON file with modality latent dimensions")
    parser.add_argument("--modality_by_modality", action="store_true", help="Process one modality at a time")
    parser.add_argument("--ignore_gene_masks", action="store_true", help="Ignore gene masks")

    # Performance arguments
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--extreme_memory_efficient", action="store_true", help="Use extreme memory efficiency techniques")
    parser.add_argument("--cuda_alloc_conf", type=str, default="max_split_size_mb:128", help="CUDA memory allocation configuration")

    # Gradient accumulation arguments
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating weights")
    parser.add_argument("--micro_batch_size", type=int, default=0, help="Size of micro-batches for gradient accumulation (0 means auto-calculate)")

    args = parser.parse_args()

    # Run training
    run_integrated_training_improved(args)



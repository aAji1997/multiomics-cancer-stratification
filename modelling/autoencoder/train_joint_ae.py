import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import time # Import time
from datetime import datetime
from tqdm.auto import tqdm  # Add tqdm import
import scipy.sparse as sp
import joblib
import json # Import json for parsing argument

# Local imports
from model import JointAutoencoder 
from data_utils import load_prepared_data, JointOmicsDataset, prepare_graph_data

# weights and biases
import wandb
import weave

# load api key from file
with open('.api_config.json', 'r') as f:
    config = json.load(f)
    WANDB_API_KEY = config['wandb_api_key']

# add to environment variables
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

def prune_graph(adj_matrix, threshold=0.5, keep_top_percent=None, min_edges_per_node=2):
    """
    Prunes the graph by either applying a threshold or keeping only a percentage of strongest edges.
    
    Args:
        adj_matrix: scipy sparse or numpy adjacency matrix
        threshold: Edge weight threshold, edges below this are pruned
        keep_top_percent: If provided, keep this percentage of strongest edges
        min_edges_per_node: Ensure each node has at least this many edges
        
    Returns:
        Pruned adjacency matrix (same format as input)
    """
    print(f"Pruning graph with initial edges: {np.sum(adj_matrix > 0)}")
    
    # Convert to scipy sparse if not already
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    
    if keep_top_percent is not None:
        # Keep top X% of edges based on weight
        flattened = adj_matrix.data.copy()
        if len(flattened) > 0:  # Only if there are non-zero elements
            cutoff_idx = max(1, int((1 - keep_top_percent) * len(flattened)))
            cutoff_value = np.sort(flattened)[cutoff_idx]
            pruned_adj = adj_matrix.copy()
            pruned_adj.data[pruned_adj.data < cutoff_value] = 0
            pruned_adj.eliminate_zeros()
        else:
            pruned_adj = adj_matrix.copy()
    else:
        # Apply threshold
        pruned_adj = adj_matrix.copy()
        pruned_adj.data[pruned_adj.data < threshold] = 0
        pruned_adj.eliminate_zeros()
    
    # Ensure each node has at least min_edges_per_node connections
    if min_edges_per_node > 0:
        n_nodes = pruned_adj.shape[0]
        for i in range(n_nodes):
            row = pruned_adj.getrow(i)
            if row.nnz < min_edges_per_node:
                # If not enough edges, add back the strongest ones from original
                if adj_matrix.getrow(i).nnz > 0:
                    row_orig = adj_matrix.getrow(i)
                    # Find missing edges to add
                    n_to_add = min_edges_per_node - row.nnz
                    if n_to_add > 0 and row_orig.nnz > row.nnz:
                        indices = row_orig.indices[row_orig.data.argsort()[::-1]]
                        # Add edges not already in pruned adj
                        added = 0
                        for idx in indices:
                            if pruned_adj[i, idx] == 0 and i != idx:  # Skip self-loops
                                pruned_adj[i, idx] = adj_matrix[i, idx]
                                added += 1
                                if added >= n_to_add:
                                    break
    
    # Make symmetric if the original was symmetric
    if (adj_matrix != adj_matrix.transpose()).nnz == 0:
        pruned_adj = pruned_adj.maximum(pruned_adj.transpose())
    
    print(f"After pruning: {pruned_adj.nnz} edges remaining ({pruned_adj.nnz/adj_matrix.nnz:.2%} of original)")
    return pruned_adj

@weave.op() # Decorate the main training function for Weave tracking
def train_joint_autoencoder(args):
    """Trains the Joint Autoencoder with WandB and Weave logging."""

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Weights & Biases / Weave Setup ---
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.cancer_type}_{run_timestamp}"
    project_name = f"joint-ae-training-{args.cancer_type}" # Customize as needed

    print("Initializing Weights & Biases and Weave...")
    try:
        # Login to wandb (use environment variable WANDB_API_KEY or run wandb login)
        # wandb.login(key=WANDB_API_KEY) # Uncomment if key is not set as env var

        # Initialize wandb run
        wandb.init(
            project=project_name,
            name=run_name,
            config=vars(args) # Log hyperparameters
        )
        
        # Initialize Weave tracking, linked to the wandb project
        # weave.init(project_name) # Weave seems to automatically use the wandb run context now
        
        print(f"W&B Run URL: {wandb.run.get_url()}")
        
    except Exception as e:
        print(f"Error initializing W&B/Weave: {e}. Proceeding without W&B tracking.")
        # Fallback to disabled mode if initialization fails
        os.environ["WANDB_DISABLED"] = "true" 
        wandb.init(mode="disabled") 
        # Weave tracking will also likely not work if wandb init fails


    # --- Data Loading and Preparation ---
    print("Loading data...")
    prepared_data = load_prepared_data(args.data_path)
    if not prepared_data or args.cancer_type not in prepared_data:
        print(f"Error: Could not load or find data for {args.cancer_type} in {args.data_path}")
        return

    cancer_data = prepared_data[args.cancer_type]
    omics_data_dict = cancer_data['omics_data']
    adj_matrix = cancer_data['adj_matrix']
    gene_list = cancer_data['gene_list']
    num_genes = len(gene_list)
    
    # ADDED: Make adjacency matrix symmetric (undirected) to match GCN assumptions
    print("Making adjacency matrix symmetric (undirected)...")
    if sp.issparse(adj_matrix):
        orig_edges = adj_matrix.nnz
        adj_matrix = adj_matrix.maximum(adj_matrix.transpose())
        sym_edges = adj_matrix.nnz
        print(f"Symmetrizing: {orig_edges} edges → {sym_edges} edges")
    else:  # If it's a dense numpy array
        orig_edges = np.sum(adj_matrix > 0)
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        sym_edges = np.sum(adj_matrix > 0)
        print(f"Symmetrizing: {orig_edges} edges → {sym_edges} edges")
    
    # ADDED: Remove self-loops from adjacency matrix
    print("Removing self-loops from adjacency matrix...")
    if sp.issparse(adj_matrix):
        num_self_loops = adj_matrix.diagonal().sum()
        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()  # Important for sparse matrices
        print(f"Removed {int(num_self_loops)} self-loops")
    else:
        num_self_loops = np.sum(np.diag(adj_matrix) > 0)
        np.fill_diagonal(adj_matrix, 0)
        print(f"Removed {int(num_self_loops)} self-loops")
    
    # Prune the graph if enabled
    orig_edges = 0
    pruned_edges = 0
    if args.prune_graph:
        print(f"Pruning graph before training...")
        orig_adj = adj_matrix.copy() if not sp.issparse(adj_matrix) else adj_matrix.copy()
        adj_matrix = prune_graph(
            adj_matrix, 
            threshold=args.prune_threshold,
            keep_top_percent=args.keep_top_percent,
            min_edges_per_node=args.min_edges_per_node
        )
        
        # Log graph statistics to WandB right after pruning
        if sp.issparse(orig_adj):
            orig_edges = orig_adj.nnz
        else:
            orig_edges = np.sum(orig_adj > 0)
            
        if sp.issparse(adj_matrix):
            pruned_edges = adj_matrix.nnz
        else:
            pruned_edges = np.sum(adj_matrix > 0)
            
        wandb.log({
            'Graph/Original_Edges': orig_edges,
            'Graph/Pruned_Edges': pruned_edges,
            'Graph/Edge_Retention_Ratio': pruned_edges / orig_edges if orig_edges > 0 else 0
        }, commit=False) # Commit later with other step 0 metrics or first batch log

    # --- Prepare Data for Model ---
    print("Converting adjacency matrix to binary for BCE target...")
    # Ensure the target for BCE loss is binary (0 or 1), handles sparse/dense
    if sp.issparse(adj_matrix):
        adj_matrix_binary_target = adj_matrix.astype(bool).astype(float) 
    else: # Handle numpy array case
        adj_matrix_binary_target = (adj_matrix > 0).astype(float)
    
    print("Preparing graph data based on --node_init_modality...")
    # Decide how to call prepare_graph_data based on the argument
    if args.node_init_modality == 'identity':
        graph_node_features, graph_edge_index, graph_edge_weight, graph_adj_tensor = prepare_graph_data(
            adj_matrix_binary_target, 
            node_init_modality='identity'
        )
    else:
        # Pass necessary omics data and gene list for non-identity features
        graph_node_features, graph_edge_index, graph_edge_weight, graph_adj_tensor = prepare_graph_data(
            adj_matrix_binary_target, 
            gene_list=gene_list,
            omics_data_dict=omics_data_dict, # Pass the full dictionary
            node_init_modality=args.node_init_modality
        )
        
    # Get the actual feature dimension from the created features
    graph_feature_dim = graph_node_features.shape[1]
    print(f"Graph node feature dimension set to: {graph_feature_dim}")

    # Move static graph data to device
    graph_node_features = graph_node_features.to(device)
    graph_edge_index = graph_edge_index.to(device)
    graph_edge_weight = graph_edge_weight.to(device) # Move edge weights to device
    graph_adj_tensor = graph_adj_tensor.to(device) # Target for graph reconstruction loss

    # --- Calculate BCE pos_weight for graph loss (Moved earlier) ---
    num_nodes = graph_adj_tensor.shape[0]
    num_possible_edges = num_nodes * num_nodes # Include self-loops for calculation simplicity
    num_positives = torch.sum(graph_adj_tensor).item()
    num_negatives = num_possible_edges - num_positives
    
    if num_positives == 0:
        print("Warning: No positive edges found in the graph adjacency tensor. Setting pos_weight to 1.")
        base_pos_weight = 1.0
    else:
        base_pos_weight = num_negatives / num_positives
        
    final_pos_weight = base_pos_weight * args.bce_pos_weight_factor
    # BCELoss expects pos_weight on the same device as the input/target tensors
    pos_weight_tensor = torch.tensor([final_pos_weight], device=device)
    print(f"Calculated BCE pos_weight: {base_pos_weight:.4f} * factor {args.bce_pos_weight_factor} = {final_pos_weight:.4f}")
    # ---------------------------------------------

    print("Creating JointOmicsDataset...")
    modalities_to_use = args.modalities.split(',') if args.modalities else ['rnaseq', 'methylation', 'scnv', 'miRNA']
    try:
        joint_omics_dataset = JointOmicsDataset(omics_data_dict, gene_list, modalities=modalities_to_use)
    except (ValueError, KeyError) as e:
        print(f"Error creating JointOmicsDataset: {e}")
        return

    if len(joint_omics_dataset) == 0:
        print("Error: JointOmicsDataset is empty.")
        return

    num_modalities = joint_omics_dataset.num_modalities
    dataloader = DataLoader(joint_omics_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)
    print(f"Created DataLoader with {len(dataloader)} batches.")

    # --- Model Instantiation ---
    # Parse modality latent dimensions from JSON file
    try:
        with open(args.modality_latents_path, 'r') as f:
            modality_latent_dims = json.load(f)
        # Basic validation: check if it's a dict and values are int
        if not isinstance(modality_latent_dims, dict) or not all(isinstance(v, int) for v in modality_latent_dims.values()):
            raise ValueError("JSON file must contain a dictionary mapping modality names to integer dimensions.")
        # Check if keys match the requested modalities
        if set(modality_latent_dims.keys()) != set(modalities_to_use):
            raise ValueError(f"Keys in modality_latents file {list(modality_latent_dims.keys())} must match --modalities {modalities_to_use}")
        print(f"Using modality latent dimensions: {modality_latent_dims}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing modality_latents JSON file: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Could not find the modality_latents JSON file at {args.modality_latents_path}")
        return
    
    # Use modalities_to_use as the order, consistent with JointOmicsDataset
    modality_order = modalities_to_use

    model = JointAutoencoder(
        num_nodes=num_genes,
        # Pass modality dims and order instead of num_modalities
        modality_latent_dims=modality_latent_dims,
        modality_order=modality_order,
        graph_feature_dim=graph_feature_dim,
        gene_embedding_dim=args.gene_embedding_dim,
        patient_embedding_dim=args.patient_embedding_dim,
        graph_dropout=args.graph_dropout
    ).to(device)

    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    wandb.summary['Model_Architecture_String'] = str(model) # Log model string representation
    wandb.summary['Total_Trainable_Parameters'] = total_params # Log total params to summary

    # --- Watch Model with WandB (Optional) ---
    # Monitors gradients and parameters. Can be resource-intensive.
    wandb.watch(model, log="gradients", log_freq=args.log_interval * len(dataloader), log_graph=True)

    # --- Loss and Optimizer ---
    # Omics Reconstruction Loss 
    omics_loss_fn = F.mse_loss # Using functional version

    # Graph Reconstruction Loss 
    graph_loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_tensor)

    # Separate learning rates if specified
    graph_ae_params = model.graph_autoencoder.parameters()
    omics_processor_params = model.omics_processor.parameters()
    
    graph_ae_lr = args.graph_ae_lr if args.graph_ae_lr is not None else args.learning_rate
    omics_processor_lr = args.omics_processor_lr if args.omics_processor_lr is not None else args.learning_rate
    
    print(f"Using LR for Graph AE: {graph_ae_lr}")
    print(f"Using LR for Omics Processor: {omics_processor_lr}")

    optimizer = optim.Adam([
        {'params': graph_ae_params, 'lr': graph_ae_lr},
        {'params': omics_processor_params, 'lr': omics_processor_lr}
    ], weight_decay=args.weight_decay) # General LR is now ignored here, set per group

    # --- Training Loop ---
    print("\nStarting joint training...")
    global_step = 0
    start_time_total = time.time()
    model.train()
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(args.epochs), desc='Training Epochs', position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        total_loss = 0.0
        total_omics_loss = 0.0
        total_graph_loss = 0.0
        total_kl_loss = 0.0

        # Create batch progress bar
        batch_pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                         desc=f'Epoch {epoch+1}/{args.epochs}', 
                         leave=False, position=1)

        for batch_idx, omics_batch_structured in batch_pbar:
            omics_batch_structured = omics_batch_structured.to(device)
            # Model forward pass now returns mu and log_var for graph embeddings
            omics_reconstructed, adj_reconstructed, z_patient, z_gene, mu, log_var = model(
                graph_node_features, graph_edge_index, omics_batch_structured,
                graph_edge_weight=graph_edge_weight
            )

            # Calculate reconstruction losses
            loss_o = omics_loss_fn(omics_reconstructed, omics_batch_structured)
            # Apply positive weight to BCE loss for graph reconstruction
            # Weight is now part of the graph_loss_fn instance
            loss_g = graph_loss_fn(adj_reconstructed, graph_adj_tensor)
            
            # Calculate KL divergence loss for the graph VGAE
            # Normalize by number of nodes (genes)
            loss_kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
            # Alternatively, normalize by num_nodes * batch_size if needed, but mean over nodes seems standard
            # loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / model.num_nodes

            # Combine losses
            combined_loss = (args.omics_loss_weight * loss_o + 
                             args.graph_loss_weight * loss_g + 
                             args.kl_loss_weight * loss_kl)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'loss': f'{combined_loss.item():.4f}', 'kl_loss': f'{loss_kl.item():.4f}'})

            # Log batch losses to WandB
            wandb.log({
                'Loss/Batch/Total': combined_loss.item(),
                'Loss/Batch/Omics': loss_o.item(),
                'Loss/Batch/Graph': loss_g.item(),
                'Loss/Batch/KL': loss_kl.item()
            }, step=global_step) # Use global_step for x-axis

            total_loss += combined_loss.item()
            total_omics_loss += loss_o.item()
            total_graph_loss += loss_g.item()
            # We also need to track total KL loss for epoch average
            total_kl_loss += loss_kl.item()
            global_step += 1
            
        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        avg_omics_loss = total_omics_loss / len(dataloader)
        avg_graph_loss = total_graph_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader) # Calculate average KL loss
        epoch_duration = time.time() - epoch_start_time

        # Update epoch progress bar with average losses
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'omics_loss': f'{avg_omics_loss:.4f}',
            'graph_loss': f'{avg_graph_loss:.4f}',
            'kl_loss': f'{avg_kl_loss:.4f}' # Add KL loss to progress bar
        })

        # Log epoch losses to WandB (use epoch as step)
        wandb.log({
            'Loss/Epoch/Total': avg_loss,
            'Loss/Epoch/Omics': avg_omics_loss,
            'Loss/Epoch/Graph': avg_graph_loss,
            'Loss/Epoch/KL': avg_kl_loss,
            'Timing/Epoch_Duration_sec': epoch_duration,
            'Training/Learning_Rate_GraphAE': optimizer.param_groups[0]['lr'],
            'Training/Learning_Rate_OmicsProc': optimizer.param_groups[1]['lr'],
            'epoch': epoch # Explicitly log epoch number for easier filtering/grouping in W&B UI
        }) # W&B automatically uses its internal step counter if 'step' isn't provided, 
           # but logging 'epoch' explicitly is good practice.

        if (epoch + 1) % args.log_interval == 0:
            print(f"\nEpoch [{epoch+1}/{args.epochs}], Avg Total Loss: {avg_loss:.6f}, "
                  f"Avg Omics Loss: {avg_omics_loss:.6f}, Avg Graph Loss: {avg_graph_loss:.6f}, "
                  f"Avg KL Loss: {avg_kl_loss:.6f}, Duration: {epoch_duration:.2f} sec") # Add KL loss to printout

    total_training_time = time.time() - start_time_total
    print(f"\nTraining finished. Total duration: {total_training_time:.2f} sec")
    # Log total training time to WandB summary
    wandb.summary['Timing/Total_Training_Duration_sec'] = total_training_time 

    # Close TensorBoard writer # REMOVED
    # writer.close() # REMOVED

    # --- Saving & WandB Artifact Logging --- #
    model_save_path = None
    embeddings_save_path = None
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save the trained joint model state dictionary
        model_save_path = os.path.join(args.output_dir, f'joint_ae_model_{args.cancer_type}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Joint model state dict saved to {model_save_path}")

        # Save the final latent embeddings (run inference)
        model.eval()
        all_patient_embeddings = []
        final_gene_embeddings = None
        with torch.no_grad():
            # Get final gene embeddings (use the mean `mu` from VGAE)
            # Pass edge_weight here as well
            # encode now returns mu, log_var, z_gene. We want mu for inference.
            mu_final, _, _ = model.graph_autoencoder.encode(
                graph_node_features, graph_edge_index, edge_weight=graph_edge_weight
            )
            final_gene_embeddings = mu_final.cpu().numpy()

            # Get patient embeddings (process data in batches)
            inference_dataloader = DataLoader(joint_omics_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            for omics_batch_structured in inference_dataloader:
                omics_batch_structured = omics_batch_structured.to(device)
                # Need z_gene (actually mu during inference) on the correct device for the encode step
                # Recalculate mu_device for consistency during inference
                mu_device, _, z_gene_eval = model.graph_autoencoder.encode( # Use z_gene_eval which is mu when model.eval()
                    graph_node_features, graph_edge_index, edge_weight=graph_edge_weight
                )
                # Pass the deterministic embedding (mu) to the omics encoder during inference
                patient_embeddings = model.omics_processor.encode(omics_batch_structured, z_gene_eval)
                all_patient_embeddings.append(patient_embeddings.cpu().numpy())

        final_patient_embeddings = np.concatenate(all_patient_embeddings, axis=0)

        # Create a dictionary with all embeddings and IDs
        embeddings_dict = {
            'gene_embeddings': final_gene_embeddings,
            'patient_embeddings': final_patient_embeddings,
            'patient_ids': np.array(joint_omics_dataset.patient_ids),
            'gene_list': gene_list
        }
        
        # Save the dictionary as a single joblib file
        embeddings_save_path = os.path.join(args.output_dir, f'joint_ae_embeddings_{args.cancer_type}.joblib')
        joblib.dump(embeddings_dict, embeddings_save_path)
        print(f"All embeddings and IDs saved to {embeddings_save_path}")

    # --- Finish Logging & Save Artifacts ---
    print("Finishing W&B run and saving artifacts...")
    if wandb.run and wandb.run.mode != "disabled": # Check if wandb was initialized successfully and not disabled
        # Save model artifact to W&B
        if model_save_path and os.path.exists(model_save_path):
            model_artifact_name = f'joint-ae-model-{args.cancer_type}-{run_timestamp}'
            artifact = wandb.Artifact(model_artifact_name, type='model', description=f"Trained Joint AE model for {args.cancer_type}")
            artifact.add_file(model_save_path)
            wandb.log_artifact(artifact)
            print(f"Model artifact '{model_artifact_name}' saved to W&B")

        # Save embeddings artifact
        if embeddings_save_path and os.path.exists(embeddings_save_path):
            embeddings_artifact_name = f'joint-ae-embeddings-{args.cancer_type}-{run_timestamp}'
            embeddings_artifact = wandb.Artifact(embeddings_artifact_name, type='embeddings', description=f"Generated embeddings for {args.cancer_type}")
            embeddings_artifact.add_file(embeddings_save_path)
            wandb.log_artifact(embeddings_artifact)
            print(f"Embeddings artifact '{embeddings_artifact_name}' saved to W&B")

        # Finish Weave and W&B runs
        # weave.finish() # finish() may not be needed if using wandb context
        wandb.finish()
    else:
        print("WandB logging was disabled or failed to initialize. Skipping artifact saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Joint Graph-Omics Autoencoder')

    # Data and Paths
    parser.add_argument('--data_path', type=str, default='data/prepared_data_both.joblib',
                        help='Path to the prepared data joblib file relative to workspace root')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type to train on')
    parser.add_argument('--modalities', type=str, default='rnaseq,methylation,scnv,miRNA',
                        help='Comma-separated list of omics modalities to use (determines input tensor order)')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models and embeddings')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='Directory to save TensorBoard logs (No longer used if W&B is enabled)')

    # Model Hyperparameters
    parser.add_argument('--gene_embedding_dim', type=int, default=64,
                        help='Dimension of the latent gene embeddings (Z_gene)')
    parser.add_argument('--patient_embedding_dim', type=int, default=128,
                        help='Dimension of the latent patient embeddings (z_p)')
    parser.add_argument('--modality_latents_path', type=str, default='config/modality_latents.json',
                       help='Path to a JSON file defining latent dimensions for each modality')
    parser.add_argument('--graph_dropout', type=float, default=0.5,
                        help='Dropout rate for GCN layers in graph AE')
    parser.add_argument('--node_init_modality', type=str, default='identity',
                       help='Method for initializing graph node features. Options: identity, or an omics type like rnaseq or methylation.')

    # Graph Pruning Parameters
    parser.add_argument('--prune_graph', action='store_true',
                       help='Whether to prune the graph before training')
    parser.add_argument('--prune_threshold', type=float, default=0.5,
                       help='Edge weight threshold for pruning (edges below this are removed)')
    parser.add_argument('--keep_top_percent', type=float, default=0.1,
                       help='Keep only this percentage of strongest edges (0.1 = top 10%)')
    parser.add_argument('--min_edges_per_node', type=int, default=2,
                       help='Ensure each node has at least this many edges after pruning')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Default learning rate for the optimizer (used if component-specific LRs are not set)')
    parser.add_argument('--graph_ae_lr', type=float, default=None,
                        help='Specific learning rate for the graph autoencoder (defaults to --learning_rate)')
    parser.add_argument('--omics_processor_lr', type=float, default=None,
                        help='Specific learning rate for the omics processor (defaults to --learning_rate)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--omics_loss_weight', type=float, default=1.0,
                        help='Weight for the omics reconstruction loss')
    parser.add_argument('--graph_loss_weight', type=float, default=0.5,
                        help='Weight for the graph reconstruction loss')
    parser.add_argument('--kl_loss_weight', type=float, default=0.01,
                       help='Weight for the KL divergence loss in VGAE')
    parser.add_argument('--bce_pos_weight_factor', type=float, default=1.0,
                       help='Factor to multiply the calculated BCE pos_weight for graph edges.')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Log training status every n epochs')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers for DataLoader')

    args = parser.parse_args()

    # Create output and log directories if they don't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


    train_joint_autoencoder(args) 
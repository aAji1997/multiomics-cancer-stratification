import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
import joblib
import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import scipy.sparse as sp
import torch.nn as nn

import hdbscan

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from modelling.gcn.hetero_gcn_model import HeteroGCN 

# Add wandb imports
import wandb

# --- Simple Decoder for Embedding Reconstruction ---
class EmbeddingDecoder(nn.Module):
    def __init__(self, gcn_out_channels, initial_patient_dim, initial_gene_dim):
        super().__init__()
        # Separate decoders for patient and gene embeddings
        self.patient_decoder = nn.Sequential(
            nn.Linear(gcn_out_channels, (gcn_out_channels + initial_patient_dim) // 2),
            nn.ReLU(),
            nn.Linear((gcn_out_channels + initial_patient_dim) // 2, initial_patient_dim)
        )
        self.gene_decoder = nn.Sequential(
            nn.Linear(gcn_out_channels, (gcn_out_channels + initial_gene_dim) // 2),
            nn.ReLU(),
            nn.Linear((gcn_out_channels + initial_gene_dim) // 2, initial_gene_dim)
        )

    def forward(self, final_patient_embeddings, final_gene_embeddings):
        rec_patient = self.patient_decoder(final_patient_embeddings)
        rec_gene = self.gene_decoder(final_gene_embeddings)
        return rec_patient, rec_gene
# --------------------------------------------------

# Load API key from file
try:
    with open('.api_config.json', 'r') as f:
        config = json.load(f)
        WANDB_API_KEY = config['wandb_api_key']
    # Add to environment variables
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
except FileNotFoundError:
    print("Warning: .api_config.json not found. W&B tracking may not work properly.")
except json.JSONDecodeError:
    print("Warning: Could not parse .api_config.json. W&B tracking may not work properly.")
except KeyError:
    print("Warning: wandb_api_key not found in .api_config.json. W&B tracking may not work properly.")
except Exception as e:
    print(f"Warning: Error loading W&B API key: {e}. W&B tracking may not work properly.")

def load_omics_for_links(omics_data_path, cancer_type, modality='rnaseq'):
    """Loads a specific omics modality for creating patient-gene links."""
    try:
        data = joblib.load(omics_data_path)
        if cancer_type not in data:
            raise KeyError(f"Cancer type '{cancer_type}' not found.")
        if modality not in data[cancer_type]['omics_data']:
             raise KeyError(f"Omics modality '{modality}' not found for {cancer_type}.")
        omics_df = data[cancer_type]['omics_data'][modality]
        # Ensure patient_id is index
        if 'patient_id' in omics_df.columns:
             omics_df = omics_df.set_index('patient_id')
        elif omics_df.index.name != 'patient_id':
             print(f"Warning: Assuming index of {modality} is patient ID.")
        print(f"Loaded {modality} data for link creation: {omics_df.shape}")
        return omics_df
    except FileNotFoundError:
        print(f"Error: Omics data file not found at {omics_data_path}")
        return None
    except KeyError as e:
         print(f"Error loading omics data: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading omics data: {e}")
        return None

def create_patient_gene_edges(omics_df, patient_ids, gene_list, link_type='threshold', threshold=0.5, top_k=None):
    """ 
    Creates patient-gene edges based on specified criteria.

    Args:
        omics_df (pd.DataFrame): DataFrame (patients x genes) for the chosen modality.
        patient_ids (list): Ordered list of patient IDs.
        gene_list (list): Ordered list of gene IDs.
        link_type (str): Method to create links ('threshold', 'top_k_per_patient', 'top_k_per_gene').
        threshold (float): Value threshold for 'threshold' type.
        top_k (int): Number of top connections for 'top_k' types.

    Returns:
        torch.Tensor: Edge index tensor (2, num_edges) for patient-gene links.
    """
    if omics_df is None:
        return torch.empty((2, 0), dtype=torch.long)

    # Align DataFrame
    try:
        omics_df_aligned = omics_df.loc[patient_ids, gene_list]
    except KeyError as e:
         print(f"Error aligning omics data: Mismatch between provided patient/gene lists and omics DataFrame index/columns. Details: {e}")
         return torch.empty((2, 0), dtype=torch.long)
    
    num_patients = len(patient_ids)
    num_genes = len(gene_list)

    source_indices = []
    target_indices = []

    print(f"Creating patient-gene links using type: '{link_type}'")

    if link_type == 'threshold':
        # Ensure threshold comparison works even if omics_df contains NaNs
        adj_matrix = (omics_df_aligned.fillna(-np.inf) >= threshold).astype(int)
        rows, cols = np.where(adj_matrix == 1)
        source_indices = rows.tolist() # Patient indices (correspond to patient_ids order)
        target_indices = cols.tolist() # Gene indices (correspond to gene_list order)

    elif link_type == 'top_k_per_patient' and top_k is not None:
        if top_k <= 0:
             print("Warning: top_k must be positive. No edges created.")
        elif top_k >= num_genes:
             print(f"Warning: top_k ({top_k}) >= num_genes ({num_genes}). Linking all non-NaN values per patient.")
             adj_matrix = (~omics_df_aligned.isna()).astype(int)
             rows, cols = np.where(adj_matrix == 1)
             source_indices = rows.tolist()
             target_indices = cols.tolist()
        else:
            for p_idx in range(num_patients):
                patient_values = omics_df_aligned.iloc[p_idx].values
                valid_indices = np.where(~np.isnan(patient_values))[0]
                if len(valid_indices) == 0:
                    continue # Skip patient if all values are NaN
                
                # Consider only top_k among the valid indices
                num_valid_to_consider = min(len(valid_indices), top_k)
                if num_valid_to_consider == 0: continue
                
                # Get indices of the top k valid values
                # Use partition to find kth largest element efficiently, then take top k
                # This avoids sorting the whole array
                kth_largest_idx = len(valid_indices) - num_valid_to_consider
                partitioned_indices = valid_indices[np.argpartition(patient_values[valid_indices], kth_largest_idx)]
                top_k_indices = partitioned_indices[kth_largest_idx:]
                
                # Double check we have the right number in case of ties exactly at threshold with partition
                if len(top_k_indices) > num_valid_to_consider:
                    # Resolve ties if partition gave more than k (take highest values among ties)
                     top_k_indices = top_k_indices[np.argsort(-patient_values[top_k_indices])[:num_valid_to_consider]]
                elif len(top_k_indices) < num_valid_to_consider:
                     # Should not happen with argpartition logic unless all values are same/NaN
                     # Fallback: if fewer than k found (e.g., many NaNs), take all valid
                     top_k_indices = valid_indices[np.argsort(patient_values[valid_indices])[-num_valid_to_consider:]]

                source_indices.extend([p_idx] * len(top_k_indices))
                target_indices.extend(top_k_indices.tolist())
                
    elif link_type == 'top_k_per_gene' and top_k is not None:
        if top_k <= 0:
             print("Warning: top_k must be positive. No edges created.")
        elif top_k >= num_patients:
             print(f"Warning: top_k ({top_k}) >= num_patients ({num_patients}). Linking all non-NaN values per gene.")
             adj_matrix = (~omics_df_aligned.isna()).astype(int)
             rows, cols = np.where(adj_matrix == 1)
             source_indices = rows.tolist()
             target_indices = cols.tolist()
        else:
             for g_idx in range(num_genes):
                 gene_values = omics_df_aligned.iloc[:, g_idx].values
                 valid_indices = np.where(~np.isnan(gene_values))[0]
                 if len(valid_indices) == 0:
                     continue
                 
                 num_valid_to_consider = min(len(valid_indices), top_k)
                 if num_valid_to_consider == 0: continue

                 kth_largest_idx = len(valid_indices) - num_valid_to_consider
                 partitioned_indices = valid_indices[np.argpartition(gene_values[valid_indices], kth_largest_idx)]
                 top_k_indices = partitioned_indices[kth_largest_idx:]

                 if len(top_k_indices) > num_valid_to_consider:
                     top_k_indices = top_k_indices[np.argsort(-gene_values[top_k_indices])[:num_valid_to_consider]]
                 elif len(top_k_indices) < num_valid_to_consider:
                      top_k_indices = valid_indices[np.argsort(gene_values[valid_indices])[-num_valid_to_consider:]]

                 source_indices.extend(top_k_indices.tolist())
                 target_indices.extend([g_idx] * len(top_k_indices))
    else:
        print(f"Warning: Invalid link_type ('{link_type}') or invalid/missing top_k. Using threshold={threshold}.")
        return create_patient_gene_edges(omics_df, patient_ids, gene_list, link_type='threshold', threshold=threshold)

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    print(f"Created {edge_index.shape[1]} patient-gene edges ('patient' -> 'gene').")
    return edge_index

def create_patient_gene_edges_from_embeddings(patient_embeddings, gene_embeddings, patient_ids, gene_list, 
                                             link_type='top_k_per_patient', top_k=10, similarity_metric='cosine', threshold=0.5):
    """
    Creates patient-gene edges based on embedding similarity when omics data isn't available.
    
    This is useful when working with precomputed embeddings from an autoencoder but no raw omics data.
    
    Args:
        patient_embeddings (torch.Tensor): Patient embeddings tensor (num_patients x embedding_dim)
        gene_embeddings (torch.Tensor): Gene embeddings tensor (num_genes x embedding_dim)
        patient_ids (list): Ordered list of patient IDs
        gene_list (list): Ordered list of gene IDs
        link_type (str): Method to create links ('top_k_per_patient', 'top_k_per_gene', 'threshold')
        top_k (int): Number of top connections for 'top_k' types
        similarity_metric (str): Method to calculate similarity ('cosine', 'euclidean', 'dot')
        threshold (float): Similarity threshold for 'threshold' link type
        
    Returns:
        torch.Tensor: Edge index tensor (2, num_edges) for patient-gene links
    """
    print(f"Creating patient-gene links from embeddings using type: '{link_type}' and {similarity_metric} similarity")
    num_patients = len(patient_ids)
    num_genes = len(gene_list)
    
    # Normalize embeddings for cosine similarity
    if similarity_metric == 'cosine':
        patient_embeddings_norm = F.normalize(patient_embeddings, p=2, dim=1)
        gene_embeddings_norm = F.normalize(gene_embeddings, p=2, dim=1)
    else:
        patient_embeddings_norm = patient_embeddings
        gene_embeddings_norm = gene_embeddings
    
    source_indices = []
    target_indices = []
    
    # Calculate similarity matrix based on chosen metric
    # Computing full similarity matrix can be memory-intensive for large graphs
    # For large graphs, consider calculating similarities in batches
    if similarity_metric == 'cosine':
        # Cosine similarity: dot product of normalized vectors
        similarity_matrix = torch.mm(patient_embeddings_norm, gene_embeddings_norm.t())
    elif similarity_metric == 'euclidean':
        # Negative euclidean distance (higher value = more similar)
        similarity_matrix = -torch.cdist(patient_embeddings, gene_embeddings, p=2)
    elif similarity_metric == 'dot':
        # Dot product (no normalization)
        similarity_matrix = torch.mm(patient_embeddings, gene_embeddings.t())
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    
    if link_type == 'threshold':
        # Connect if similarity exceeds threshold
        rows, cols = torch.where(similarity_matrix >= threshold)
        source_indices = rows.tolist()
        target_indices = cols.tolist()
    
    elif link_type == 'top_k_per_patient':
        if top_k <= 0:
            print("Warning: top_k must be positive. No edges created.")
        elif top_k >= num_genes:
            print(f"Warning: top_k ({top_k}) >= num_genes ({num_genes}). Creating all possible connections.")
            # Create all possible connections
            for p_idx in range(num_patients):
                source_indices.extend([p_idx] * num_genes)
                target_indices.extend(list(range(num_genes)))
        else:
            # For each patient, find top_k most similar genes
            for p_idx in range(num_patients):
                similarities = similarity_matrix[p_idx]
                # Get indices of top_k highest similarities
                top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()
                source_indices.extend([p_idx] * len(top_k_indices))
                target_indices.extend(top_k_indices)
    
    elif link_type == 'top_k_per_gene':
        if top_k <= 0:
            print("Warning: top_k must be positive. No edges created.")
        elif top_k >= num_patients:
            print(f"Warning: top_k ({top_k}) >= num_patients ({num_patients}). Creating all possible connections.")
            # Create all possible connections
            for g_idx in range(num_genes):
                source_indices.extend(list(range(num_patients)))
                target_indices.extend([g_idx] * num_patients)
        else:
            # For each gene, find top_k most similar patients
            for g_idx in range(num_genes):
                similarities = similarity_matrix[:, g_idx]
                # Get indices of top_k highest similarities
                top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()
                source_indices.extend(top_k_indices)
                target_indices.extend([g_idx] * len(top_k_indices))
    
    else:
        print(f"Warning: Invalid link_type '{link_type}'. Using top_k_per_patient with k={top_k}.")
        return create_patient_gene_edges_from_embeddings(
            patient_embeddings, gene_embeddings, patient_ids, gene_list, 
            link_type='top_k_per_patient', top_k=top_k
        )
    
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    print(f"Created {edge_index.shape[1]} patient-gene edges from embeddings ('patient' -> 'gene').")
    return edge_index

def contrastive_loss(z1, z2, temperature=0.1, pos_mask=None, neg_mask=None):
    """ 
    Calculates InfoNCE contrastive loss between two sets of embeddings.
    Assumes z1 and z2 are embeddings for the *same set of nodes* under different augmentations/views.
    Args:
        z1 (Tensor): Embeddings from view 1 (N x Dim).
        z2 (Tensor): Embeddings from view 2 (N x Dim).
        temperature (float): Temperature scaling factor.
        pos_mask (Tensor, optional): Boolean mask for positive pairs (N x N). Defaults to identity.
        neg_mask (Tensor, optional): Boolean mask for negative pairs (N x N). Defaults to all non-positive.

    Returns:
        Tensor: Scalar contrastive loss value.
    """
    n_nodes = z1.shape[0]
    device = z1.device

    # Normalize embeddings
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Similarity matrix
    sim = torch.matmul(z1, z2.t()) / temperature # N x N

    if pos_mask is None:
        # Default: Positive pairs are the node itself in the other view (diagonal)
        pos_mask = torch.eye(n_nodes, dtype=torch.bool, device=device)
    if neg_mask is None:
        # Default: Negative pairs are all pairs except the positive ones
        neg_mask = ~pos_mask

    # Calculate exp(sim) for positives and negatives
    exp_sim = torch.exp(sim)
    
    # Numerator: sum of exp(sim) for positive pairs (typically just the diagonal element)
    # Ensure pos_sim has size N, even if some rows have no positive examples (shouldn't happen with default mask)
    pos_sim = torch.where(pos_mask, exp_sim, torch.zeros_like(exp_sim)).sum(dim=1) # N
    
    # Denominator: sum of exp(sim) for all valid pairs (positive + negative)
    # Ensure neg_sim has size N
    neg_sim = torch.where(neg_mask, exp_sim, torch.zeros_like(exp_sim)).sum(dim=1) # N
    denominator = pos_sim + neg_sim

    # Calculate loss per node: -log(pos / denominator)
    # Handle cases where denominator is zero (no positive or negative examples found for a node)
    loss_per_node = -torch.log(pos_sim / (denominator + 1e-8)) # Add epsilon for stability
    # If pos_sim is zero (e.g., no positive samples found), log(0) is -inf. Set these losses to 0.
    loss_per_node = torch.where(pos_sim > 1e-8, loss_per_node, torch.zeros_like(loss_per_node))

    return loss_per_node.mean()

def run_survival_analysis(patient_ids, clusters, clinical_df, output_dir, cancer_type):
    """Performs survival analysis using lifelines."""
    if clinical_df is None:
        print("Skipping survival analysis: Clinical data not provided or failed to load.")
        return

    print("\n--- Performing Survival Analysis ---")
    # Create DataFrame for analysis
    cluster_df = pd.DataFrame({'patient_id': patient_ids, 'cluster': clusters})
    
    # Merge with clinical data
    required_cols = ['patient_id', 'duration', 'event']
    # Standardize patient ID format if needed (e.g., TCGA barcodes)
    # Assuming clinical_df might have full barcodes, while patient_ids are shorter
    if 'bcr_patient_barcode' in clinical_df.columns and 'patient_id' not in clinical_df.columns:
         clinical_df.rename(columns={'bcr_patient_barcode': 'patient_id'}, inplace=True)
        
    # Attempt to match patient IDs flexibly (e.g., first 12 chars of TCGA barcode)
    try:
        cluster_df['patient_id_short'] = cluster_df['patient_id'].str[:12]
        clinical_df['patient_id_short'] = clinical_df['patient_id'].str[:12]
        analysis_df = pd.merge(cluster_df, clinical_df, on='patient_id_short', how='inner')
        # Use original full ID from clinical data if available
        analysis_df['patient_id'] = analysis_df['patient_id_y'] 
        analysis_df.drop(columns=['patient_id_x', 'patient_id_y', 'patient_id_short'], inplace=True)
    except Exception as e:
        print(f"Warning: Could not merge using shortened IDs ({e}). Attempting direct merge on 'patient_id'.")
        analysis_df = pd.merge(cluster_df, clinical_df, on='patient_id', how='inner')

    # Check required columns after merge
    if not all(col in analysis_df.columns for col in required_cols):
        print(f"Error: Merged clinical data must contain columns: {required_cols}")
        print(f"Found columns after merge: {list(analysis_df.columns)}")
        return
        
    # Ensure duration and event are numeric and drop rows with NaNs in essential columns
    analysis_df['duration'] = pd.to_numeric(analysis_df['duration'], errors='coerce')
    analysis_df['event'] = pd.to_numeric(analysis_df['event'], errors='coerce')
    analysis_df = analysis_df.dropna(subset=required_cols)
    analysis_df['event'] = analysis_df['event'].astype(int) # Ensure event is integer
    
    if analysis_df.empty:
        print("Error: No matching patients found with valid survival data after merging.")
        return
        
    print(f"Survival analysis on {len(analysis_df)} patients.")
    num_unique_clusters = analysis_df['cluster'].nunique()

    # Log-rank test (only if more than one cluster)
    p_value = None
    if num_unique_clusters > 1:
        try:
            results = multivariate_logrank_test(analysis_df['duration'], analysis_df['cluster'], analysis_df['event'])
            p_value = results.p_value
            print(f"Multivariate Log-rank Test p-value: {p_value:.4f}")
            # Save p-value
            with open(os.path.join(output_dir, f'logrank_pvalue_{cancer_type}.txt'), 'w') as f:
                f.write(f'{p_value:.6f}')
            
        except Exception as e:
            print(f"Error during log-rank test: {e}")
    else:
         print("Skipping log-rank test (only one cluster).")

    # Kaplan-Meier plots
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    for cluster_label in sorted(analysis_df['cluster'].unique()):
        subset = analysis_df[analysis_df['cluster'] == cluster_label]
        kmf.fit(subset['duration'], event_observed=subset['event'], label=f'Cluster {cluster_label} (n={len(subset)})')
        kmf.plot_survival_function(ax=ax)

    plt.title(f'Kaplan-Meier Survival Curves by Cluster ({cancer_type})')
    plt.xlabel("Time (days)") # Adjust label if duration unit is different
    plt.ylabel("Survival Probability")
    # Add p-value to plot if calculated
    if p_value is not None:
         # Position text carefully
         plt.text(0.95, 0.05, f'Log-rank p={p_value:.3f}', 
                  transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', 
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    plt.legend(title="Cluster")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, f'kaplan_meier_{cancer_type}.png')
    plt.savefig(plot_save_path)
    print(f"Kaplan-Meier plot saved to {plot_save_path}")
    plt.close()

# --- Graph Augmentation Helpers --- #

def augment_features(x_dict, mask_rate):
    """Applies feature masking to node features."""
    if mask_rate == 0.0:
        return x_dict # No augmentation
    augmented_x_dict = {}
    for node_type, x in x_dict.items():
        mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate)).to(x.device)
        augmented_x_dict[node_type] = x * mask
    return augmented_x_dict

def augment_edges(edge_index_dict, drop_rate):
    """Applies edge dropping to graph edges."""
    if drop_rate == 0.0:
        return edge_index_dict # No augmentation
    augmented_edge_index_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        if edge_index.numel() > 0: # Check if there are any edges of this type
            num_edges = edge_index.shape[1]
            keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
            augmented_edge_index_dict[edge_type] = edge_index[:, keep_mask]
        else:
            augmented_edge_index_dict[edge_type] = edge_index # Keep empty edge index
    return augmented_edge_index_dict

# --- Main Execution Function --- #

def run_stratification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()

    # --- Initialize Weights & Biases --- #
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"gcn-{args.cancer_type}-{args.gcn_conv_type}-l{args.gcn_layers}-{run_timestamp}"
    project_name = f"hetero-gcn-{args.cancer_type}"  # Customize project name as needed
    
    # Derive default clinical data path from cancer type if not explicitly provided
    if args.clinical_data_path is None:
        args.clinical_data_path = f"data/{args.cancer_type}/omics_data/clinical.csv"
        print(f"Clinical data path not specified, using default: {args.clinical_data_path}")
    
    print("Initializing Weights & Biases...")
    try:
        # Initialize wandb run
        wandb.init(
            project=project_name,
            name=run_name,
            config=vars(args)  # Log all arguments as config
        )
        print(f"W&B Run URL: {wandb.run.get_url()}")
        # Initialize wandb Table for tracking graph structure
        graph_structure_table = wandb.Table(columns=["node_type", "count"])
        
    except Exception as e:
        print(f"Error initializing W&B: {e}. Proceeding without W&B tracking.")
        # Fallback to disabled mode if initialization fails
        os.environ["WANDB_DISABLED"] = "true"
        wandb.init(mode="disabled")

    # --- Load Pre-computed Embeddings --- #
    print(f"\nLoading embeddings from: {args.embedding_path}")
    try:
        embeddings_data = joblib.load(args.embedding_path)
        gene_embeddings = torch.tensor(embeddings_data['gene_embeddings'], dtype=torch.float32)
        patient_embeddings = torch.tensor(embeddings_data['patient_embeddings'], dtype=torch.float32)
        patient_ids = list(np.array(embeddings_data['patient_ids'])) 
        gene_list = list(np.array(embeddings_data['gene_list']))  
        print(f"Loaded {len(patient_ids)} patients and {len(gene_list)} genes.")
        
        # Print embedding dimensions for clarity
        print(f"Loaded gene embeddings with shape: {gene_embeddings.shape}")
        print(f"Loaded patient embeddings with shape: {patient_embeddings.shape}")
        gene_embeddings.requires_grad_(False)
        # Log embedding info to W&B
        if wandb.run and wandb.run.mode != "disabled":
            wandb.log({
                "Data/Num_Patients": len(patient_ids),
                "Data/Num_Genes": len(gene_list),
                "Data/Gene_Embedding_Dim": gene_embeddings.shape[1],
                "Data/Patient_Embedding_Dim": patient_embeddings.shape[1]
            })
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {args.embedding_path}")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # --- Load Necessary Data for Edges --- #
    print(f"\nLoading original data from: {args.original_data_path}")
    adj_matrix_gene_gene = None
    omics_df_for_links = None
    gene_masks = None  # Initialize gene masks to None
    try:
        prepared_data = joblib.load(args.original_data_path)
        if args.cancer_type not in prepared_data:
             raise KeyError(f"Cancer type '{args.cancer_type}' not found in {args.original_data_path}.")
        cancer_data = prepared_data[args.cancer_type]
        
        # Extract gene masks if available
        if 'gene_masks' in cancer_data:
            gene_masks = cancer_data['gene_masks']
            print(f"Loaded gene masks for modalities: {list(gene_masks.keys())}")
            
            # Log mask statistics to W&B
            if wandb.run and wandb.run.mode != "disabled":
                mask_stats_table = wandb.Table(columns=["Modality", "Originally Present Genes", "Total Genes", "Presence Ratio"])
                for modality, mask in gene_masks.items():
                    present_count = sum(mask)
                    total_count = len(mask)
                    presence_ratio = present_count / total_count
                    mask_stats_table.add_data(modality, present_count, total_count, f"{presence_ratio:.2%}")
                    wandb.log({
                        f"Masks/{modality}/Present_Count": present_count,
                        f"Masks/{modality}/Total_Count": total_count,
                        f"Masks/{modality}/Presence_Ratio": presence_ratio
                    })
                wandb.log({"Masks/Statistics": mask_stats_table})
                
            # Validate gene masks length against gene list
            for modality, mask in gene_masks.items():
                if len(mask) != len(gene_list):
                    print(f"Warning: Gene mask for {modality} has length {len(mask)}, but gene list has length {len(gene_list)}.")
                    print("Masks will be ignored due to mismatched dimensions. Make sure masks and gene list align.")
                    gene_masks = None
                    break
        else:
            print("No gene masks found in the data. Proceeding without mask-aware aggregation.")
        
        # Get gene interaction graph
        if 'adj_matrix' not in cancer_data:
             raise KeyError("'adj_matrix' (gene interactions) not found in prepared data.")
        adj_matrix_gene_gene = cancer_data['adj_matrix'] 
        
        # Make adjacency matrix symmetric (undirected) to match GCN assumptions
        print("Making adjacency matrix symmetric (undirected)...")
        if sp.issparse(adj_matrix_gene_gene):
            orig_edges = adj_matrix_gene_gene.nnz
            adj_matrix_gene_gene = adj_matrix_gene_gene.maximum(adj_matrix_gene_gene.transpose())
            sym_edges = adj_matrix_gene_gene.nnz
            print(f"Symmetrizing: {orig_edges} edges → {sym_edges} edges")
        else:  # If it's a dense numpy array
            orig_edges = np.sum(adj_matrix_gene_gene > 0)
            adj_matrix_gene_gene = np.maximum(adj_matrix_gene_gene, adj_matrix_gene_gene.T)
            sym_edges = np.sum(adj_matrix_gene_gene > 0)
            print(f"Symmetrizing: {orig_edges} edges → {sym_edges} edges")
        
        # Remove self-loops from adjacency matrix
        print("Removing self-loops from adjacency matrix...")
        if sp.issparse(adj_matrix_gene_gene):
            num_self_loops = adj_matrix_gene_gene.diagonal().sum()
            adj_matrix_gene_gene.setdiag(0)
            adj_matrix_gene_gene.eliminate_zeros()  # Important for sparse matrices
            print(f"Removed {int(num_self_loops)} self-loops")
        else:
            num_self_loops = np.sum(np.diag(adj_matrix_gene_gene) > 0)
            np.fill_diagonal(adj_matrix_gene_gene, 0)
            print(f"Removed {int(num_self_loops)} self-loops")
        
        # Check gene list consistency
        if 'gene_list' not in cancer_data:
             raise KeyError("'gene_list' not found in prepared data.")
        original_gene_list = list(np.array(cancer_data['gene_list']))
        if original_gene_list != gene_list:
            print("Critical Warning: Gene lists from original data and embeddings file do NOT match!")
            print("This implies the adjacency matrix and omics data might not align with the loaded gene embeddings.")
            if args.force_gene_list_alignment:
                print("Attempting to reindex the adjacency matrix to match embedding gene list order...")
                # This could be implemented to map between the two gene lists, but would be complex
                # For now, we'll just warn and proceed assuming the match is close enough
            print("Attempting to proceed assuming adj_matrix rows/cols correspond to embedding gene_list order, but results may be incorrect.")
            # Check dimensions as a basic safeguard
            if adj_matrix_gene_gene.shape[0] != len(gene_list):
                 print(f"Error: Adjacency matrix dimension ({adj_matrix_gene_gene.shape[0]}) mismatch with embedding gene list ({len(gene_list)}). Cannot proceed.")
                 return
        
        # Load specific omics for patient-gene links
        if args.pg_link_omics:
            omics_df_for_links = load_omics_for_links(args.original_data_path, args.cancer_type, args.pg_link_omics)
            if omics_df_for_links is None:
                 print("Warning: Failed to load omics data for patient-gene links. Links will not be created.")
        else:
            print("No omics modality specified for patient-gene links. Using precomputed embeddings only.")

    except FileNotFoundError:
        print(f"Error: Original data file not found at {args.original_data_path}")
        return
    except KeyError as e:
         print(f"Error accessing data in original file: {e}")
         return
    except Exception as e:
        print(f"Error loading original data: {e}")
        return
        
    # --- Load Clinical Data --- #
    clinical_df = None
    if args.clinical_data_path:
         print(f"\nLoading clinical data from: {args.clinical_data_path}")
         if not os.path.exists(args.clinical_data_path):
             print(f"Warning: Clinical data file not found at {args.clinical_data_path}")
             print("Survival analysis will be skipped.")
         else:
             try:
                 # Assuming CSV/TSV format, adjust as needed
                 sep = '	' if args.clinical_data_path.endswith('.tsv') else ','
                 clinical_df = pd.read_csv(args.clinical_data_path, sep=sep)
                 print(f"Loaded clinical data for {len(clinical_df)} records initially.")
             except Exception as e:
                 print(f"Error loading or processing clinical data: {e}")
                 clinical_df = None # Ensure it's None if loading fails
    else:
         print("\nNo clinical data path provided. Survival analysis will be skipped.")

    # --- Construct HeteroData Object --- #
    print("\nConstructing heterogeneous graph...")
    hetero_data = HeteroData()

    # Node Features
    hetero_data['patient'].x = patient_embeddings
    hetero_data['gene'].x = gene_embeddings
    hetero_data['patient'].node_ids = patient_ids # Store original IDs
    hetero_data['gene'].node_ids = gene_list

    # Gene-Gene Edges (Require adj_matrix_gene_gene to be loaded)
    if adj_matrix_gene_gene is not None:
        if hasattr(adj_matrix_gene_gene, "tocoo"): 
            coo = adj_matrix_gene_gene.tocoo()
            edge_index_gg = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        else: 
            try:
                # Attempt conversion assuming numpy array or compatible
                adj_tensor_gg = torch.tensor(adj_matrix_gene_gene, dtype=torch.float32)
                edge_index_gg, _ = dense_to_sparse(adj_tensor_gg)
            except Exception as e:
                 print(f"Error converting gene adjacency matrix to sparse format: {e}. Skipping gene-gene edges.")
                 edge_index_gg = torch.empty((2,0), dtype=torch.long)
        
        hetero_data['gene', 'interacts', 'gene'].edge_index = edge_index_gg
        print(f"Added {edge_index_gg.shape[1]} gene-gene ('interacts') edges.")
    else:
        print("Skipping gene-gene edges as adjacency matrix was not loaded.")
        hetero_data['gene', 'interacts', 'gene'].edge_index = torch.empty((2,0), dtype=torch.long)

    # Patient-Gene Edges
    if omics_df_for_links is not None:
        # Create edges from omics data if available
        edge_index_pg = create_patient_gene_edges(omics_df_for_links, patient_ids, gene_list, 
                                                  link_type=args.pg_link_type, 
                                                  threshold=args.pg_link_threshold, 
                                                  top_k=args.pg_link_top_k)
        hetero_data['patient', 'expresses', 'gene'].edge_index = edge_index_pg
        
        # Add Reverse Edges for Message Passing
        if edge_index_pg.numel() > 0: 
            edge_index_gp = edge_index_pg[[1, 0], :] 
            hetero_data['gene', 'rev_expresses', 'patient'].edge_index = edge_index_gp
            print(f"Added {edge_index_gp.shape[1]} reverse patient-gene ('rev_expresses') edges.")
        else:
            hetero_data['gene', 'rev_expresses', 'patient'].edge_index = torch.empty((2,0), dtype=torch.long)
            print("No reverse patient-gene edges added (no forward edges found).")
    else:
        # Create edges based on embedding similarity when omics data isn't available
        print("Creating patient-gene links based on embedding similarity...")
        edge_index_pg = create_patient_gene_edges_from_embeddings(
            patient_embeddings, 
            gene_embeddings, 
            patient_ids, 
            gene_list,
            link_type=args.pg_link_type if args.pg_link_type in ['top_k_per_patient', 'top_k_per_gene', 'threshold'] else 'top_k_per_patient',
            top_k=args.pg_link_top_k if args.pg_link_top_k else 10,
            similarity_metric=args.embedding_similarity_metric if hasattr(args, 'embedding_similarity_metric') else 'cosine',
            threshold=args.pg_link_threshold
        )
        hetero_data['patient', 'expresses', 'gene'].edge_index = edge_index_pg
        
        # Add Reverse Edges for Message Passing
        if edge_index_pg.numel() > 0: 
            edge_index_gp = edge_index_pg[[1, 0], :] 
            hetero_data['gene', 'rev_expresses', 'patient'].edge_index = edge_index_gp
            print(f"Added {edge_index_gp.shape[1]} reverse patient-gene ('rev_expresses') edges.")
        else:
            hetero_data['gene', 'rev_expresses', 'patient'].edge_index = torch.empty((2,0), dtype=torch.long)
            print("No reverse patient-gene edges added (no forward edges found).")

    # Move data to device before model instantiation
    try:
        hetero_data = hetero_data.to(device)
        print(f"Moved HeteroData to {device}.")
    except Exception as e:
        print(f"Error moving HeteroData to device {device}: {e}")
        return

    print("\nHeteroData structure:")
    print(hetero_data)
    try:
        hetero_data.validate(raise_on_error=True)
        print("Heterogeneous graph validation successful.")
    except Exception as e:
        print(f"Graph validation failed: {e}")
        # Decide whether to proceed with potentially invalid graph
        if args.force_proceed_on_validation_error:
             print("Warning: Proceeding despite graph validation failure due to --force_proceed flag.")
        else:
             return

    # --- Prepare Targets for Reconstruction Losses ---
    # Keep initial embeddings for reconstruction target
    initial_patient_embeddings = patient_embeddings.to(device)
    initial_gene_embeddings = gene_embeddings.to(device)

    # Prepare binary graph adjacency target tensor
    if adj_matrix_gene_gene is not None:
        if sp.issparse(adj_matrix_gene_gene):
            adj_matrix_binary_target = adj_matrix_gene_gene.astype(bool).astype(float)
        else: # Handle numpy array case
            adj_matrix_binary_target = (adj_matrix_gene_gene > 0).astype(float)
        graph_adj_tensor = torch.tensor(adj_matrix_binary_target.todense() if sp.issparse(adj_matrix_binary_target) else adj_matrix_binary_target, dtype=torch.float32).to(device)
    else:
        graph_adj_tensor = None

    # Calculate BCE pos_weight for graph loss
    pos_weight_tensor = None
    if graph_adj_tensor is not None:
        num_nodes = graph_adj_tensor.shape[0]
        num_possible_edges = num_nodes * num_nodes
        num_positives = torch.sum(graph_adj_tensor).item()
        num_negatives = num_possible_edges - num_positives
        if num_positives == 0:
            base_pos_weight = 1.0
        else:
            base_pos_weight = num_negatives / num_positives
        # No separate factor argument here, use a fixed reasonable factor or assume 1.0
        # Adjust factor here if needed, e.g., 0.5 or 2.0, based on initial tests
        final_pos_weight = base_pos_weight * 1.0 
        pos_weight_tensor = torch.tensor([final_pos_weight], device=device)
        print(f"Calculated BCE pos_weight for graph loss: {final_pos_weight:.4f}")
    # ---------------------------------------------

    # After graph construction, log graph structure metrics to W&B
    if wandb.run and wandb.run.mode != "disabled":
        # Log node and edge counts
        for node_type in hetero_data.node_types:
            count = hetero_data[node_type].num_nodes
            wandb.log({f"Graph/Nodes/{node_type}": count})
            graph_structure_table.add_data(node_type, count)
            
        # Log edge counts for each edge type
        for edge_type in hetero_data.edge_types:
            edge_count = hetero_data[edge_type].num_edges
            src, relation, dst = edge_type
            edge_name = f"{src}_{relation}_{dst}"
            wandb.log({f"Graph/Edges/{edge_name}": edge_count})
            
        # Log graph structure table
        wandb.log({"Graph/Structure": graph_structure_table})

    # --- Model Instantiation --- #
    metadata = hetero_data.metadata()
    node_feature_dims = {ntype: hetero_data[ntype].x.shape[1] for ntype in metadata[0]}
    model = HeteroGCN(
        metadata=metadata,
        node_feature_dims=node_feature_dims,
        hidden_channels=args.gcn_hidden_dim,
        out_channels=args.gcn_output_dim, 
        num_layers=args.gcn_layers,
        conv_type=args.gcn_conv_type,
        num_heads=args.gcn_gat_heads,
        dropout_rate=args.gcn_dropout,
        use_layer_norm=not args.gcn_no_norm, # Use norm unless flag is set
        gene_masks=gene_masks if not args.ignore_gene_masks else None # Pass gene masks if available and not ignored
    ).to(device)

    # --- Initialize Lazy Modules --- #
    print("\nInitializing lazy modules...")
    with torch.no_grad():
        try:
            # Use the actual data for initialization
            _ = model(hetero_data.x_dict, hetero_data.edge_index_dict)
            print("Lazy modules initialized successfully.")
        except Exception as e:
            print(f"Error during dummy forward pass for initialization: {e}")
            print("Proceeding, but model parameters might not be fully initialized yet.")
    # --- End Initialization --- #

    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Log model architecture to W&B
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Model/Total_Parameters": total_params,
            "Model/Architecture_String": str(model),
            "Model/Using_Gene_Masks": gene_masks is not None and not args.ignore_gene_masks
        })
        # Watch model for gradient and weight tracking (optional)
        wandb.watch(model, log="all", log_freq=10)

    # --- Unsupervised GCN Training --- #
    training_losses = []
    # Always train the model in this script
    if args.train_gcn_epochs > 0:
        print(f"\n--- Training GCN for {args.train_gcn_epochs} epochs ---")
        print(f"Using precomputed embeddings as node features (gene: {initial_gene_embeddings.shape[1]}, patient: {initial_patient_embeddings.shape[1]} dimensions)")
        
        # Instantiate losses
        graph_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if graph_adj_tensor is not None else None
        embedding_loss_fn = torch.nn.MSELoss()
        
        # Instantiate Embedding Decoder if weight > 0
        embedding_decoder = None
        if args.embedding_loss_weight > 0:
            embedding_decoder = EmbeddingDecoder(
                gcn_out_channels=args.gcn_output_dim,
                initial_patient_dim=initial_patient_embeddings.shape[1],
                initial_gene_dim=initial_gene_embeddings.shape[1]
            ).to(device)
            # Combine parameters for the optimizer
            combined_params = list(model.parameters()) + list(embedding_decoder.parameters())
        else:
            combined_params = model.parameters()
            
        optimizer = optim.Adam(combined_params, lr=args.gcn_lr, weight_decay=args.gcn_weight_decay)
        
        model.train() # Set model to training mode
        if embedding_decoder: embedding_decoder.train()
        
        train_pbar = tqdm(range(args.train_gcn_epochs), desc='GCN Training Epochs')

        for epoch in train_pbar:
            epoch_start_time = time.time()
            optimizer.zero_grad()

            # --- Graph Augmentation --- #
            # Create two augmented views
            x_dict_view1 = augment_features(hetero_data.x_dict, args.aug_feature_mask_rate)
            edge_index_dict_view1 = augment_edges(hetero_data.edge_index_dict, args.aug_edge_drop_rate)

            x_dict_view2 = augment_features(hetero_data.x_dict, args.aug_feature_mask_rate)
            edge_index_dict_view2 = augment_edges(hetero_data.edge_index_dict, args.aug_edge_drop_rate)
            # --- End Augmentation --- #

            # Forward pass for both views
            z_dict_view1 = model(x_dict_view1, edge_index_dict_view1)
            z_dict_view2 = model(x_dict_view2, edge_index_dict_view2)

            # --- Contrastive Loss (Patients) ---
            z_patient_view1 = z_dict_view1.get('patient')
            z_patient_view2 = z_dict_view2.get('patient')

            if z_patient_view1 is None or z_patient_view2 is None or z_patient_view1.shape[0] == 0:
                print(f"Warning: Patient embeddings missing or empty at epoch {epoch}. Skipping contrastive loss.")
                loss_contrastive = torch.tensor(0.0, device=device)
            else:
                loss_contrastive = contrastive_loss(z_patient_view1, z_patient_view2, temperature=args.contrastive_temp)

            # --- Graph Reconstruction Loss (Genes - use view 1) ---
            loss_graph = torch.tensor(0.0, device=device)
            if args.graph_loss_weight > 0 and graph_loss_fn is not None:
                z_gene_view1 = z_dict_view1.get('gene')
                if z_gene_view1 is not None:
                    adj_rec_logits = z_gene_view1 @ z_gene_view1.t()
                    loss_graph = graph_loss_fn(adj_rec_logits, graph_adj_tensor)
                else:
                    print(f"Warning: Gene embeddings missing at epoch {epoch}. Skipping graph loss.")

            # --- Embedding Reconstruction Loss (use view 1) ---
            # This loss attempts to reconstruct the *initial* AE embeddings from the *final* GCN embeddings.
            # Even though initial_gene_embeddings are frozen, this loss component can still be trained.
            # It encourages the GCN's output gene embeddings (z_gene_view1) to be decodable back to the frozen initial ones,
            # potentially acting as a regularizer.
            loss_embedding = torch.tensor(0.0, device=device)
            if args.embedding_loss_weight > 0 and embedding_decoder is not None:
                z_patient_view1 = z_dict_view1.get('patient') # Re-get in case None earlier
                z_gene_view1 = z_dict_view1.get('gene') # Re-get in case None earlier
                if z_patient_view1 is not None and z_gene_view1 is not None:
                    rec_patient, rec_gene = embedding_decoder(z_patient_view1, z_gene_view1)
                    loss_emb_patient = embedding_loss_fn(rec_patient, initial_patient_embeddings)
                    loss_emb_gene = embedding_loss_fn(rec_gene, initial_gene_embeddings)
                    loss_embedding = loss_emb_patient + loss_emb_gene # Simple sum, could weight
                else:
                    print(f"Warning: Final patient or gene embeddings missing at epoch {epoch}. Skipping embedding loss.")

            # --- Combine Losses ---
            # MODIFIED: Exclude graph_loss from total_loss calculation as gene embeddings are frozen.
            # Graph loss is still calculated above for logging purposes.
            total_loss = (loss_contrastive +
                          # args.graph_loss_weight * loss_graph + # excluded
                          args.embedding_loss_weight * loss_embedding)

           
            if epoch == 0:
                print("Note: Initial gene embeddings are frozen. Graph reconstruction loss is calculated for logging but excluded from training gradients.")
                print(f"Training objective: contrastive_loss + {args.embedding_loss_weight} * embedding_loss")

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                 print(f"Warning: NaN or Inf loss detected at epoch {epoch}. Skipping backward pass.")
                 loss_item = np.nan # Record NaN loss
                 graph_loss_item = loss_graph.item() if isinstance(loss_graph, torch.Tensor) else 0.0
                 embedding_loss_item = loss_embedding.item() if isinstance(loss_embedding, torch.Tensor) else 0.0
            elif total_loss.requires_grad:
                 total_loss.backward()
                 # Gradient clipping
                 torch.nn.utils.clip_grad_norm_(combined_params, max_norm=1.0)
                 optimizer.step()
                 loss_item = total_loss.item()
                 graph_loss_item = loss_graph.item()
                 embedding_loss_item = loss_embedding.item()
            else: # Loss does not require grad
                loss_item = total_loss.item()
                graph_loss_item = loss_graph.item()
                embedding_loss_item = loss_embedding.item()

            epoch_duration = time.time() - epoch_start_time
            training_losses.append(loss_item)
            train_pbar.set_postfix({'total_loss': f'{loss_item:.4f}',
                                  'graph_loss': f'{graph_loss_item:.4f}',
                                  'emb_loss': f'{embedding_loss_item:.4f}'})

            # Log metrics to W&B
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({
                    "Training/Total_Loss": loss_item,
                    "Training/Contrastive_Loss": loss_contrastive.item(), # Log original contrastive loss
                    "Training/Graph_Reconstruction_Loss": graph_loss_item,
                    "Training/Embedding_Reconstruction_Loss": embedding_loss_item,
                    "Training/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Training/Epoch_Duration_sec": epoch_duration,
                    "epoch": epoch
                })

        print(f"Finished GCN training. Final total loss: {training_losses[-1]:.4f}")
        # Plot training loss
        if training_losses:
            plt.figure(figsize=(8, 4))
            plt.plot(training_losses)
            plt.title("GCN Hybrid Training Loss") # Updated title
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.grid(True)
            loss_plot_path = os.path.join(args.output_dir, f'gcn_hybrid_training_loss_{args.cancer_type}.png')
            plt.savefig(loss_plot_path)
            print(f"Training loss plot saved to {loss_plot_path}")
            
            # Log loss plot to W&B
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({"Training/Loss_Curve": wandb.Image(loss_plot_path)})
            
            plt.close()
    else:
        print("\nWarning: train_gcn_epochs set to 0. No training will be performed.")
        print("Consider setting a positive value for train_gcn_epochs to train the model.")

    # --- Save Final Model and Embeddings --- #
    if args.output_dir:
        # Ensure directory exists (already created in main)
        print("\n--- Saving Trained Model and Embeddings ---")

        # Save GCN model state dictionary
        model_save_path = os.path.join(args.output_dir, f'hetero_gcn_model_{args.cancer_type}.pth')
        try:
             torch.save(model.state_dict(), model_save_path)
             print(f"HeteroGCN model state dict saved to {model_save_path}")
             
             # Log model as artifact to W&B
             if wandb.run and wandb.run.mode != "disabled":
                 model_artifact = wandb.Artifact(
                     name=f"hetero-gcn-model-{args.cancer_type}",
                     type="model",
                     description=f"Trained HeteroGCN model for {args.cancer_type}"
                 )
                 model_artifact.add_file(model_save_path)
                 wandb.log_artifact(model_artifact)
        except Exception as e:
             print(f"Error saving model state dict: {e}")

        # Save Embedding Decoder state dictionary if it exists
        if embedding_decoder is not None:
            decoder_save_path = os.path.join(args.output_dir, f'embedding_decoder_{args.cancer_type}.pth')
            try:
                 torch.save(embedding_decoder.state_dict(), decoder_save_path)
                 print(f"Embedding Decoder state dict saved to {decoder_save_path}")
                 # Optionally log decoder artifact to W&B
                 if wandb.run and wandb.run.mode != "disabled":
                     decoder_artifact = wandb.Artifact(
                         name=f"embedding-decoder-{args.cancer_type}",
                         type="decoder",
                         description=f"Embedding decoder for {args.cancer_type}"
                     )
                     decoder_artifact.add_file(decoder_save_path)
                     wandb.log_artifact(decoder_artifact)
            except Exception as e:
                 print(f"Error saving embedding decoder state dict: {e}")

        # Generate and save embeddings (use non-augmented data)
        model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            final_embeddings_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
            
        if 'patient' not in final_embeddings_dict:
            print("Error: 'patient' embeddings not found in final GCN output.")
        else:
            final_patient_embeddings_gcn = final_embeddings_dict['patient'].cpu().numpy()
            print(f"Final patient embeddings shape after GCN: {final_patient_embeddings_gcn.shape}")
            
            # Extract gene embeddings as well
            final_gene_embeddings_gcn = None
            if 'gene' in final_embeddings_dict:
                final_gene_embeddings_gcn = final_embeddings_dict['gene'].cpu().numpy()
                print(f"Final gene embeddings shape after GCN: {final_gene_embeddings_gcn.shape}")
            else:
                print("Warning: 'gene' embeddings not found in final GCN output.")
            
            # Log embedding dimensionality to W&B
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({"Embeddings/Final_Patient_Embedding_Dim": final_patient_embeddings_gcn.shape[1]})
                if final_gene_embeddings_gcn is not None:
                    wandb.log({"Embeddings/Final_Gene_Embedding_Dim": final_gene_embeddings_gcn.shape[1]})
                
                # Log embedding visualization if not too many patients
                if len(patient_ids) <= 1000:
                    try:
                        # Create PCA plot of embeddings
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        embeddings_2d = pca.fit_transform(final_patient_embeddings_gcn)
                        
                        plt.figure(figsize=(10, 8))
                        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
                        plt.title(f"PCA of GCN Patient Embeddings ({args.cancer_type})")
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.tight_layout()
                        pca_plot_path = os.path.join(args.output_dir, f'patient_embeddings_pca_{args.cancer_type}.png')
                        plt.savefig(pca_plot_path)
                        plt.close()
                        
                        wandb.log({"Embeddings/PCA_Plot": wandb.Image(pca_plot_path)})
                    except Exception as e:
                        print(f"Warning: Could not create embedding visualization: {e}")
            
            results = {
                'patient_ids': patient_ids,
                'final_patient_embeddings_gcn': final_patient_embeddings_gcn,
                'gene_list': gene_list,
                'gene_embeddings': final_gene_embeddings_gcn,  # Save gene embeddings
                'gcn_training_losses': training_losses,
                'args': vars(args) # Save final args dict
            }
            embeddings_save_path = os.path.join(args.output_dir, f'gcn_embeddings_{args.cancer_type}.joblib')
            try:
                 joblib.dump(results, embeddings_save_path)
                 print(f"GCN embeddings saved to {embeddings_save_path}")
                 
                 # Log embeddings as artifact to W&B
                 if wandb.run and wandb.run.mode != "disabled":
                     embeddings_artifact = wandb.Artifact(
                         name=f"gcn-embeddings-{args.cancer_type}",
                         type="embeddings",
                         description=f"GCN embeddings for {args.cancer_type} patients and genes"
                     )
                     embeddings_artifact.add_file(embeddings_save_path)
                     wandb.log_artifact(embeddings_artifact)
            except Exception as e:
                 print(f"Error saving embeddings to joblib: {e}")
        
        # Save graph structure for reference
        graph_info = {
            'node_counts': {ntype: hetero_data[ntype].x.shape[0] for ntype in metadata[0]},
            'edge_counts': {etype: hetero_data[etype].edge_index.shape[1] for etype in metadata[1]}
        }
        graph_info_path = os.path.join(args.output_dir, f'graph_structure_{args.cancer_type}.json')
        try:
            with open(graph_info_path, 'w') as f:
                json.dump(graph_info, f, indent=2)
            print(f"Graph structure information saved to {graph_info_path}")
            
            # Log graph structure as artifact to W&B
            if wandb.run and wandb.run.mode != "disabled":
                graph_structure_artifact = wandb.Artifact(
                    name=f"graph-structure-{args.cancer_type}",
                    type="graph-info",
                    description=f"Graph structure information for {args.cancer_type}"
                )
                graph_structure_artifact.add_file(graph_info_path)
                wandb.log_artifact(graph_structure_artifact)
        except Exception as e:
            print(f"Error saving graph structure: {e}")
    
    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"\nTotal execution time: {total_execution_time:.2f} seconds")
    
    # Log final summary metrics to W&B
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Summary/Total_Execution_Time_sec": total_execution_time,
            "Summary/Final_Loss": training_losses[-1] if training_losses else None,
        })
        # Finish the wandb run
        wandb.finish()
        
    print("\nTraining completed successfully!")
    print(f"For clustering and evaluation, use evaluate_risk_stratification_gcn.py with the model or embeddings from: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heterogeneous GCN for Patient Risk Stratification - Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Input Data --- #
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to .joblib file with patient/gene embeddings from AE.')
    parser.add_argument('--original_data_path', type=str, required=True,
                        help='Path to original prepared_data .joblib (for graph structure, omics links).')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type key in the prepared_data file.')
    parser.add_argument('--clinical_data_path', type=str, default=None,
                        help='Path to clinical data CSV/TSV file for survival analysis. If not specified, defaults to data/{cancer_type}/omics_data/clinical.csv')

    # --- Graph Construction --- #
    pg_group = parser.add_argument_group('Patient-Gene Link Construction')
    pg_group.add_argument('--pg_link_omics', type=str, default='rnaseq',
                         help='Omics modality to use for patient-gene links (rnaseq, methylation, etc.)')
    pg_group.add_argument('--pg_link_type', type=str, default='threshold', 
                        choices=['threshold', 'top_k_per_patient', 'top_k_per_gene'],
                        help='Method for creating patient-gene edges from the chosen omics data.')
    pg_group.add_argument('--pg_link_threshold', type=float, default=0.5,
                       help='Threshold value if pg_link_type is "threshold". Value depends on omics data scaling/meaning.')
    pg_group.add_argument('--pg_link_top_k', type=int, default=50,
                       help='Value of K if pg_link_type uses "top_k". Connects patient to top K genes or vice versa.')
    pg_group.add_argument('--embedding_similarity_metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Similarity metric to use when creating patient-gene links from embeddings.')

    # --- GCN Model --- #
    gcn_model_group = parser.add_argument_group('GCN Model Architecture')
    gcn_model_group.add_argument('--gcn_hidden_dim', type=int, default=64,
                        help='Hidden dimension for GCN layers.')
    gcn_model_group.add_argument('--gcn_output_dim', type=int, default=32,
                        help='Output dimension for final GCN embeddings (used for clustering).')
    gcn_model_group.add_argument('--gcn_layers', type=int, default=2,
                        help='Number of HeteroGCN layers.')
    gcn_model_group.add_argument('--gcn_conv_type', type=str, default='sage', choices=['gcn', 'sage', 'gat'],
                        help='Type of GNN convolution layer.')
    gcn_model_group.add_argument('--gcn_gat_heads', type=int, default=4,
                        help='Number of attention heads if using GATConv.')
    gcn_model_group.add_argument('--gcn_dropout', type=float, default=0.5,
                        help='Dropout rate in GCN layers.')
    gcn_model_group.add_argument('--gcn_no_norm', action='store_true',
                       help='Disable Layer Normalization in GCN layers.')
    gcn_model_group.add_argument('--ignore_gene_masks', action='store_true',
                       help='Ignore gene masks even if they are available in the data.')

    # --- GCN Training --- #
    gcn_train_group = parser.add_argument_group('GCN Training')
    gcn_train_group.add_argument('--train_gcn_epochs', type=int, default=100,
                        help='Number of epochs to train GCN with contrastive loss.')
    gcn_train_group.add_argument('--gcn_lr', type=float, default=0.001,
                        help='Learning rate for GCN optimizer.')
    gcn_train_group.add_argument('--gcn_weight_decay', type=float, default=1e-5,
                        help='Weight decay for GCN optimizer.')
    gcn_train_group.add_argument('--contrastive_temp', type=float, default=0.1,
                        help='Temperature for contrastive loss.')
    gcn_train_group.add_argument('--graph_loss_weight', type=float, default=0.4,
                        help='Weight for the graph reconstruction loss term.')
    gcn_train_group.add_argument('--embedding_loss_weight', type=float, default=0.2,
                        help='Weight for the initial AE embedding reconstruction loss term.')

    # --- Augmentation ---
    gcn_train_group.add_argument('--aug_feature_mask_rate', type=float, default=0.3,
                        help='Probability of masking node features during augmentation.')
    gcn_train_group.add_argument('--aug_edge_drop_rate', type=float, default=0.2,
                        help='Probability of dropping edges during augmentation.')

    # --- Output & Misc --- #
    output_group = parser.add_argument_group('Output & Miscellaneous')
    output_group.add_argument('--output_dir', type=str, default='./gcn_training_results',
                        help='Base directory to save results (timestamped subfolder will be created).')
    output_group.add_argument('--force_proceed_on_validation_error', action='store_true',
                       help='Attempt to proceed even if HeteroData validation fails.')
    output_group.add_argument('--force_gene_list_alignment', action='store_true',
                       help='Force alignment of gene lists between embeddings and original data. Use with caution.')

    args = parser.parse_args()

    # --- Create unique output directory --- #
    if args.output_dir:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add key hyperparameters to directory name for easier identification
        run_name = f"{args.cancer_type}_train_gcn{args.gcn_layers}l_{args.gcn_conv_type}_e{args.train_gcn_epochs}_{time_stamp}"
        args.output_dir = os.path.join(args.output_dir, run_name)
        try:
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Created output directory: {args.output_dir}")
        except OSError as e:
             print(f"Error creating output directory {args.output_dir}: {e}")
             # Fallback to a default name if creation fails
             args.output_dir = f"./gcn_training_results/run_{time_stamp}" 
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Using fallback output directory: {args.output_dir}")
    else:
         # Handle case where output_dir is None or empty if necessary
         print("Warning: No output directory specified. Results will not be saved.")

    # --- Run Main Function --- #
    run_stratification(args) 
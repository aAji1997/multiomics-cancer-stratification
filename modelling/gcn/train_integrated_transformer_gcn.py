# modelling/gcn/train_integrated_transformer_gcn.py
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
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import scipy.sparse as sp

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute import paths
from modelling.gcn.integrated_transformer_gcn_model import IntegratedTransformerGCN
from modelling.gcn.train_risk_stratification_gcn import contrastive_loss, create_patient_gene_edges

import wandb

# --- New Loss Functions for Reconstruction-Focused Training --- #

def structure_preservation_loss(current_embeddings, original_embeddings):
    """
    Preserves the relative distances between embeddings to maintain the structure
    of the original embeddings while allowing refinement.

    Args:
        current_embeddings: Current gene embeddings from the model
        original_embeddings: Original pre-trained gene embeddings

    Returns:
        Tensor: Loss value measuring structural difference
    """
    # Normalize embeddings
    current_norm = F.normalize(current_embeddings, p=2, dim=1)
    original_norm = F.normalize(original_embeddings, p=2, dim=1)

    # Compute pairwise similarity matrices
    current_sim = torch.mm(current_norm, current_norm.t())
    original_sim = torch.mm(original_norm, original_norm.t())

    # Loss is the difference between similarity matrices
    return F.mse_loss(current_sim, original_sim)

def gene_interaction_loss(gene_embeddings, adj_matrix, positive_weight=1.0, negative_weight=0.1):
    """
    Loss that encourages interacting genes to have similar embeddings
    and non-interacting genes to have dissimilar embeddings.

    Args:
        gene_embeddings: Gene embeddings from the model
        adj_matrix: Binary adjacency matrix of gene-gene interactions
        positive_weight: Weight for positive interaction loss
        negative_weight: Weight for negative interaction loss (non-interactions)

    Returns:
        Tensor: Loss value encouraging gene-gene interaction structure
    """
    # Compute similarity between gene embeddings
    gene_sim = torch.mm(F.normalize(gene_embeddings, p=2, dim=1),
                       F.normalize(gene_embeddings, p=2, dim=1).t())

    # Positive interactions should have high similarity
    pos_loss = -torch.mean(gene_sim * adj_matrix) * positive_weight

    # Negative interactions (non-edges) should have low similarity
    neg_mask = 1.0 - adj_matrix
    neg_loss = torch.mean(gene_sim * neg_mask) * negative_weight

    return pos_loss + neg_loss


# Load API key from file (reuse logic)
try:
    with open('.api_config.json', 'r') as f:
        config = json.load(f)
        WANDB_API_KEY = config['wandb_api_key']
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
except Exception as e:
    print(f"Warning: Error loading W&B API key: {e}. W&B tracking may not work properly.")

# --- Data Loading --- #

def load_raw_omics_data(data_path, cancer_type):
    """Loads raw omics data for multiple modalities."""
    print(f"Loading raw omics data for {cancer_type} from {data_path}...")
    try:
        prepared_data = joblib.load(data_path)
        if cancer_type not in prepared_data:
            raise KeyError(f"Cancer type '{cancer_type}' not found in {data_path}.")

        cancer_data = prepared_data[cancer_type]
        if 'omics_data' not in cancer_data or not cancer_data['omics_data']:
             raise KeyError(f"'omics_data' dictionary not found or empty for {cancer_type}.")

        raw_omics_dict = {}
        patient_ids_sets = [] # Store patient ID sets from all modalities

        # First, collect patient IDs from all modalities (excluding clinical)
        print("Collecting patient IDs from all modalities...")
        for modality, df in cancer_data['omics_data'].items():
            # Skip clinical data - only use omics modalities for initial alignment
            if modality.lower() == 'clinical':
                continue

            print(f"  Checking {modality} data for patient IDs...")
            # Ensure patient_id is index and get patient list
            if 'patient_id' in df.columns:
                current_patients = set(df['patient_id'].tolist())
            elif df.index.name == 'patient_id':
                current_patients = set(df.index.tolist())
            else:
                print(f"  Warning: Assuming index of {modality} is patient ID.")
                current_patients = set(df.index.tolist())

            print(f"  Found {len(current_patients)} patients in {modality}.")
            patient_ids_sets.append(current_patients)

        # Get intersection of all patient IDs across non-clinical modalities
        if patient_ids_sets:
            common_patients = set.intersection(*patient_ids_sets)
            patient_ids_list = sorted(list(common_patients))
            print(f"Common patients across all omics modalities: {len(patient_ids_list)}")

            if len(patient_ids_list) == 0:
                raise ValueError("No common patients found across omics modalities.")
        else:
            raise ValueError("No omics modalities found for patient ID alignment.")

        # Now load each modality and align to the common patient IDs
        for modality, df in cancer_data['omics_data'].items():
            # Skip clinical data - only use for patient ID alignment
            if modality.lower() == 'clinical':
                continue

            print(f"  Loading {modality} data...")
            # Ensure patient_id is index
            if 'patient_id' in df.columns:
                df = df.set_index('patient_id')
            elif df.index.name != 'patient_id':
                 print(f"  Warning: Assuming index of {modality} is patient ID.")

            # Align to common patient IDs
            df_aligned = df.loc[patient_ids_list]

            # Check if all columns are numeric, attempt conversion
            try:
                df_numeric = df_aligned.apply(pd.to_numeric, errors='coerce')
            except Exception as e:
                 print(f"  Warning: Could not convert all columns in {modality} to numeric ({e}). Check data types.")
                 df_numeric = df_aligned # Proceed with original types if conversion fails broadly

            # Handle potential NaNs
            if df_numeric.isnull().values.any():
                print(f"  Warning: Found NaNs in {modality} data. Filling with 0.")
                df_numeric = df_numeric.fillna(0)

            raw_omics_dict[modality] = torch.tensor(df_numeric.values, dtype=torch.float32)
            print(f"  Loaded {modality} tensor shape: {raw_omics_dict[modality].shape}")

        print(f"Finished loading raw omics data for {len(patient_ids_list)} patients.")
        return raw_omics_dict, patient_ids_list

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None
    except KeyError as e:
         print(f"Error accessing data key: {e}")
         return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading raw omics data: {e}")
        return None, None

def load_gene_embeddings(gene_embedding_path):
    """Loads pre-computed gene embeddings and corresponding gene list."""
    print(f"Loading gene embeddings from: {gene_embedding_path}")
    try:
        if gene_embedding_path.endswith('.joblib'):
            embedding_data = joblib.load(gene_embedding_path)
            # Adapt keys based on how gene embeddings were saved previously
            if 'gene_embeddings' in embedding_data and 'gene_list' in embedding_data:
                gene_embeddings = torch.tensor(embedding_data['gene_embeddings'], dtype=torch.float32)
                gene_list = list(np.array(embedding_data['gene_list']))
                print(f"Loaded gene embeddings shape: {gene_embeddings.shape}")
                print(f"Loaded gene list length: {len(gene_list)}")
                return gene_embeddings, gene_list
            else:
                raise KeyError("Required keys ('gene_embeddings', 'gene_list') not found in joblib file.")
        elif gene_embedding_path.endswith('.pt') or gene_embedding_path.endswith('.pth'):
             # Assume it's a dictionary saved with torch.save
             embedding_data = torch.load(gene_embedding_path)
             if isinstance(embedding_data, dict) and 'gene_embeddings' in embedding_data and 'gene_list' in embedding_data:
                 gene_embeddings = embedding_data['gene_embeddings'].float()
                 gene_list = embedding_data['gene_list']
                 if not isinstance(gene_list, list): gene_list = list(gene_list)
                 print(f"Loaded gene embeddings shape: {gene_embeddings.shape}")
                 print(f"Loaded gene list length: {len(gene_list)}")
                 return gene_embeddings, gene_list
             else:
                  raise ValueError("Saved PyTorch file does not contain expected dictionary with 'gene_embeddings' and 'gene_list'.")
        # Add other potential formats like .npy + separate gene list file if needed
        else:
            raise ValueError(f"Unsupported file format for gene embeddings: {gene_embedding_path}")

    except FileNotFoundError:
        print(f"Error: Gene embedding file not found at {gene_embedding_path}")
        return None, None
    except Exception as e:
        print(f"Error loading gene embeddings: {e}")
        return None, None

def load_gene_interactions_and_masks(data_path, cancer_type):
    """Loads gene interaction adjacency matrix, associated gene list, and gene masks."""
    print(f"Loading gene interactions and masks for {cancer_type} from {data_path}...")
    try:
        prepared_data = joblib.load(data_path)
        if cancer_type not in prepared_data:
            raise KeyError(f"Cancer type '{cancer_type}' not found.")

        cancer_data = prepared_data[cancer_type]
        if 'adj_matrix' not in cancer_data:
             raise KeyError("'adj_matrix' (gene interactions) not found.")
        if 'gene_list' not in cancer_data:
             raise KeyError("'gene_list' for interactions not found.")

        adj_matrix = cancer_data['adj_matrix']
        interaction_gene_list = list(np.array(cancer_data['gene_list']))

        # Make adjacency matrix symmetric (undirected) to match GCN assumptions
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

        # Remove self-loops from adjacency matrix
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

        # Load gene masks if they exist
        gene_masks = None
        if 'gene_masks' in cancer_data:
            gene_masks = cancer_data['gene_masks']
            print(f"  Loaded gene masks for modalities: {list(gene_masks.keys())}")
        else:
            print("  No gene masks found in data.")

        print(f"Loaded interaction adj matrix shape: {adj_matrix.shape}")
        print(f"Loaded interaction gene list length: {len(interaction_gene_list)}")
        return adj_matrix, interaction_gene_list, gene_masks

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None, None
    except KeyError as e:
         print(f"Error accessing data key for interactions/masks: {e}")
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred loading interactions/masks: {e}")
        return None, None, None

# --- Augmentation --- #

def augment_raw_omics(raw_omics_dict, mask_rate):
    """Applies feature masking to raw omics data tensors."""
    if mask_rate == 0.0:
        return raw_omics_dict
    augmented_dict = {}
    for omics_type, x in raw_omics_dict.items():
        mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate)).to(x.device)
        augmented_dict[omics_type] = x * mask
    return augmented_dict

def augment_edges(edge_index_dict, drop_rate):
    """Applies edge dropping to graph edges."""
    if drop_rate == 0.0:
        return edge_index_dict
    augmented_edge_index_dict = {}
    for edge_type, edge_index in edge_index_dict.items():
        if edge_index.numel() > 0:
            num_edges = edge_index.shape[1]
            keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
            augmented_edge_index_dict[edge_type] = edge_index[:, keep_mask]
        else:
            augmented_edge_index_dict[edge_type] = edge_index # Keep empty
    return augmented_edge_index_dict

# --- Main Execution --- #

def run_integrated_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()

    # --- W&B Init --- #
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Include training strategy in run name
    strategy_tag = "recon" if args.training_strategy == 'reconstruction' else "contrast"
    gene_emb_tag = "unfrozen" if args.unfreeze_gene_embeddings else "frozen"

    run_name = f"integ-{strategy_tag}-{gene_emb_tag}-{args.cancer_type}-{run_timestamp}"
    project_name = f"integrated-trans-gcn-{args.cancer_type}"

    print("Initializing Weights & Biases...")
    try:
        # Add custom tags for easier filtering
        tags = [
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
        # Use updated constructor to avoid deprecation warning
        scaler = torch.amp.GradScaler('cuda')
        # Set PyTorch to allocate memory more conservatively
        torch.cuda.empty_cache()
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = args.cuda_alloc_conf
    else:
        scaler = None

    # For any CUDA device, set the memory allocation configuration
    if device.type == 'cuda' and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = args.cuda_alloc_conf
        print(f"Setting PYTORCH_CUDA_ALLOC_CONF={args.cuda_alloc_conf}")
        torch.cuda.empty_cache()  # Clear any existing allocations

    # --- Load Data --- #
    # Load Raw Omics
    raw_omics_data_dict, patient_ids = load_raw_omics_data(args.original_data_path, args.cancer_type)
    if raw_omics_data_dict is None: return
    omics_input_dims = {k: v.shape[1] for k, v in raw_omics_data_dict.items()}
    print(f"Raw Omics Input Dims: {omics_input_dims}")

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

    num_patients = len(patient_ids)
    num_genes = len(gene_list)

    # Log data stats
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Data/Num_Patients": num_patients,
            "Data/Num_Genes": num_genes,
            "Data/Num_Omics_Types": len(raw_omics_data_dict),
            "Data/Gene_Embedding_Dim": gene_embeddings.shape[1]
        })
        for k, v in omics_input_dims.items():
            wandb.log({f"Data/InputDim_{k}": v})

        # Log mask statistics
        if gene_masks is not None:
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
        final_pos_weight = base_pos_weight * 1.0 # Adjust factor if needed
        pos_weight_tensor = torch.tensor([final_pos_weight], device=device)
        print(f"Calculated BCE pos_weight for graph loss: {final_pos_weight:.4f}")
    # Raw omics data (raw_omics_data_dict) is already loaded and on device
    # ---------------------------------------------

    # Log graph structure
    if wandb.run and wandb.run.mode != "disabled":
        graph_structure_table = wandb.Table(columns=["node_type", "count"])
        graph_structure_table.add_data("patient", num_patients)
        graph_structure_table.add_data("gene", num_genes)
        wandb.log({"Graph/Nodes/patient": num_patients, "Graph/Nodes/gene": num_genes})

        for edge_type, edges in edge_index_dict.items():
            src, rel, dst = edge_type
            edge_name = f"{src}_{rel}_{dst}"
            wandb.log({f"Graph/Edges/{edge_name}": edges.shape[1]})
        wandb.log({"Graph/Structure": graph_structure_table})

    # --- Model Instantiation --- #
    print("\nInstantiating IntegratedTransformerGCN model...")

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
        use_mixed_precision=args.use_mixed_precision
    ).to(device)

    # Store modality order for later use in loss calculation
    if add_decoder:
        args.decoder_modality_order = sorted(omics_input_dims.keys())

    # --- Prepare Omics Reconstruction Target --- #
    omics_target_tensor = None
    if add_decoder:
        # The OmicsDecoder can output in two formats based on use_modality_specific_decoders:
        # 1. A single concatenated tensor (original approach)
        # 2. A dictionary of tensors per modality + concatenated tensor

        if args.use_modality_specific_decoders:
            print(f"Preparing target tensor for modality-specific decoders")
            # For modality-specific decoders, we still need a concatenated tensor
            # but we need to be careful about the order of modalities

            # Store the modality order for reference
            modality_order = sorted(omics_input_dims.keys())
            args.decoder_modality_order = modality_order

            # Create concatenated target tensor
            target_tensors = []
            for mod_name in modality_order:
                mod_data = raw_omics_data_dict[mod_name]
                target_tensors.append(mod_data)

            # First, determine the shape needed
            batch_size = next(iter(raw_omics_data_dict.values())).shape[0]

            # Reshape and concatenate to match decoder's expected output format
            # Target should be [batch_size, num_genes, num_modalities]
            if num_genes > 0:
                # This is the ideal approach, but requires careful tensor reshaping
                # For now, simplify by using the same approach for all cases

                # Create targets with shape [batch_size, num_genes, num_modalities]
                omics_target_tensor = torch.zeros((batch_size, num_genes, len(modality_order)),
                                               dtype=torch.float32, device=device)

                # Fill in the values
                # Note: This is a simplification - in a real implementation, you need to map raw omics data
                # to the correct gene positions. For now, use the same approach as the non-modality specific case.
                for mod_idx, mod_name in enumerate(modality_order):
                    mod_data = raw_omics_data_dict[mod_name]  # [batch_size, feature_dim]

                    # For each gene position, use the corresponding feature if available
                    # or duplicate values across genes if feature_dim < num_genes
                    feature_dim = mod_data.shape[1]

                    # Approach 1: If features exactly match genes (or can be evenly divided)
                    if feature_dim == num_genes or feature_dim % num_genes == 0:
                        # Reshape to [batch_size, num_genes, feature_dim/num_genes]
                        if feature_dim == num_genes:
                            # Direct 1:1 mapping
                            omics_target_tensor[:, :, mod_idx] = mod_data
                        else:
                            # Need to handle multiple features per gene
                            # For now, use average of features per gene
                            features_per_gene = feature_dim // num_genes
                            reshaped = mod_data.reshape(batch_size, num_genes, features_per_gene)
                            # Use mean of features for each gene
                            omics_target_tensor[:, :, mod_idx] = reshaped.mean(dim=2)
                    else:
                        # Approach 2: If no direct mapping, duplicate or sample
                        print(f"Warning: Number of features ({feature_dim}) in {mod_name} doesn't align with genes ({num_genes}).")
                        print(f"Using placeholder approach - this may not reflect accurate gene-feature relationships.")
                        # Simplification: Use the same values for all genes
                        # This is NOT biologically accurate but serves as a placeholder
                        for g in range(num_genes):
                            # Assign each gene the mean value across all features
                            omics_target_tensor[:, g, mod_idx] = mod_data.mean(dim=1)

                print(f"Created modality-specific target tensor with shape: {omics_target_tensor.shape}")
                print(f"Target has {len(modality_order)} modalities in order: {modality_order}")
            else:
                print("Error: num_genes is zero or negative, cannot create target tensor")
                omics_target_tensor = None
        else:
            # Original approach: For concatenated decoder output
            print("Preparing target tensor for concatenated decoder output")

            # OmicsDecoder outputs shape [batch_size, num_genes, total_feature_dim]
            # We need to transform raw_omics_data_dict to match this format

            # First, extract necessary dimensions
            batch_size = next(iter(raw_omics_data_dict.values())).shape[0]

            # Create target tensor with correct shape
            total_feature_dim = sum(omics_input_dims.values())
            omics_target_tensor = torch.zeros((batch_size, num_genes, total_feature_dim),
                                         dtype=torch.float32, device=device)

            # Depending on how raw_omics_data is structured, we need to fill this tensor properly
            # For now, use a placeholder approach that assumes raw_omics_data maps to genes somehow

            # Store the modality order for reference (might be needed for loss calculation)
            decoder_modality_order = sorted(omics_input_dims.keys())
            args.decoder_modality_order = decoder_modality_order

            feature_start_idx = 0
            for mod_name in decoder_modality_order:
                mod_data = raw_omics_data_dict[mod_name]  # [batch_size, feature_dim]
                feature_dim = mod_data.shape[1]

                # Placeholder approach: assign each gene the same feature values
                # In a real implementation, you need a proper mapping from omics features to genes
                for g in range(num_genes):
                    omics_target_tensor[:, g, feature_start_idx:feature_start_idx+feature_dim] = mod_data

                feature_start_idx += feature_dim

            print(f"Created concatenated target tensor with shape: {omics_target_tensor.shape}")

        # Log warning about the simplifications
        print("Note: Target tensor is prepared using simplifications that may not reflect actual gene-omics relationships.")
        print("      Consider implementing a proper feature-to-gene mapping based on your data structure.")

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

    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Model/Total_Parameters": total_params,
            "Model/Using_Gene_Masks": gene_masks is not None and not args.ignore_gene_masks,
            "Model/Using_Modality_Specific_Decoders": args.use_modality_specific_decoders,
            "Model/Decoder_Activation": args.decoder_activation
        })

        # Add details about the loss functions being used
        if gene_masks is not None and not args.ignore_gene_masks and args.omics_loss_weight > 0:
            wandb.log({
                "Loss/Using_Masked_MSE": True,
                "Loss/Mask_Info": {
                    "num_modalities_with_masks": len(gene_masks),
                    "modalities": list(gene_masks.keys())
                }
            })

            # Log mask coverage for each modality
            for modality, mask in gene_masks.items():
                present_ratio = sum(mask) / len(mask) if mask else 0
                wandb.log({
                    f"Masks/{modality}/Present_Ratio": present_ratio,
                    f"Masks/{modality}/Present_Count": sum(mask) if mask else 0,
                    f"Masks/{modality}/Total_Count": len(mask) if mask else 0
                })
        else:
            wandb.log({"Loss/Using_Masked_MSE": False})
        # wandb.watch(model, log="all", log_freq=max(1, args.train_epochs // 100))

    # --- Training --- #
    if args.train_epochs <= 0:
        print("\nWarning: train_epochs set to 0. Skipping training.")
    else:
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

        model.train()
        train_pbar = tqdm(range(args.train_epochs), desc='Training Epochs')
        training_losses = []

        # Move full data to device (if not already done)
        # gene_embeddings = gene_embeddings.to(device) # Already moved
        # edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()} # Already moved
        # raw_omics_data_dict already moved

        # --- Instantiate Losses ---
        graph_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if graph_adj_tensor is not None else None
        # Use masked MSE if gene masks are available, otherwise standard MSE
        if gene_masks and not args.ignore_gene_masks:
            print("Using masked MSE for omics reconstruction")
            # Implementation for concatenated omics tensor
            def masked_omics_mse_loss(pred, target):
                """
                Calculate MSE loss with masking for originally present genes,
                adapted for concatenated omics data format.

                Args:
                    pred: Predicted values [batch, genes, concatenated_features]
                    target: Target values [batch, genes, concatenated_features]

                Returns:
                    Masked MSE loss (considers original genes only)
                """
                # Verify input dimensions
                if pred.dim() != 3 or target.dim() != 3:
                    print(f"Warning: Expected 3D tensors, got pred: {pred.dim()}D, target: {target.dim()}D")
                    return F.mse_loss(pred, target)  # Fallback to standard MSE

                batch_size, pred_genes, feature_dim = pred.shape
                target_batch_size, target_genes, target_feature_dim = target.shape

                # Check if dimensions are compatible
                if batch_size != target_batch_size or pred_genes != target_genes or feature_dim != target_feature_dim:
                    print(f"Warning: Dimension mismatch in masked MSE. Using standard MSE as fallback.")
                    print(f"  Pred shape: {pred.shape}, Target shape: {target.shape}")
                    return F.mse_loss(pred, target)  # Fallback to standard MSE

                # Create a combined tensor mask matching the input dimensions
                tensor_mask = torch.ones_like(pred)

                # Keep track of current position in the concatenated dimension
                current_pos = 0

                # Apply different mask for each modality slice
                for _, modality in enumerate(sorted(omics_input_dims.keys())):
                    if modality in gene_masks:
                        # Calculate the width of this modality's slice
                        modality_width = omics_input_dims[modality]

                        # Validate modality width against remaining feature space
                        if current_pos + modality_width > feature_dim:
                            print(f"Warning: Modality {modality} width ({modality_width}) exceeds remaining feature space. Adjusting...")
                            modality_width = feature_dim - current_pos
                            if modality_width <= 0:
                                print(f"  No space left for modality {modality}. Skipping.")
                                continue

                        # Get the binary mask for this modality
                        mod_mask = gene_masks[modality]

                        # Verify mask length matches expected genes
                        if len(mod_mask) != pred_genes:
                            print(f"Warning: Gene mask length mismatch for {modality}: "
                                  f"expected {pred_genes}, got {len(mod_mask)}. Adjusting...")
                            # Handle mismatch by truncation or padding with ones
                            if len(mod_mask) > pred_genes:
                                # Truncate
                                mod_mask = mod_mask[:pred_genes]
                            else:
                                # Pad with ones (assume unmapped genes are original)
                                mod_mask = mod_mask + [1] * (pred_genes - len(mod_mask))

                        # Convert to tensor and ensure correct type/device
                        mod_mask = torch.tensor(mod_mask, dtype=tensor_mask.dtype, device=tensor_mask.device)

                        # Reshape to [1, num_genes, 1] for broadcasting
                        mod_mask = mod_mask.view(1, -1, 1)

                        # Expand mask to match the width of this modality's slice
                        mod_mask_expanded = mod_mask.expand(batch_size, -1, modality_width)

                        # Place in the correct position in the tensor mask
                        tensor_mask[:, :, current_pos:current_pos + modality_width] = mod_mask_expanded

                        # Update position for next modality
                        current_pos += modality_width

                # Calculate MSE only on masked elements (originally present genes)
                squared_diff = (pred - target) ** 2
                masked_squared_diff = squared_diff * tensor_mask

                # Get the sum of masked values and the count of mask elements
                sum_squared_diff = masked_squared_diff.sum()
                mask_sum = tensor_mask.sum()

                # Return mean MSE over masked elements only
                if mask_sum > 0:
                    return sum_squared_diff / mask_sum
                else:
                    # Fallback if no masked elements (should never happen)
                    return F.mse_loss(pred, target)

            # Use our custom masked MSE loss
            omics_loss_fn = masked_omics_mse_loss
        else:
            print("Using standard MSE for omics reconstruction")
            omics_loss_fn = F.mse_loss

        for epoch in train_pbar:
            epoch_start_time = time.time()
            optimizer.zero_grad()

            # --- Augmentation for Contrastive Loss --- #
            # Augment raw omics data
            raw_omics_view1 = augment_raw_omics(raw_omics_data_dict, args.aug_feature_mask_rate)
            raw_omics_view2 = augment_raw_omics(raw_omics_data_dict, args.aug_feature_mask_rate)
            # Augment graph edges
            edge_index_dict_view1 = augment_edges(edge_index_dict, args.aug_edge_drop_rate)
            edge_index_dict_view2 = augment_edges(edge_index_dict, args.aug_edge_drop_rate)
            # Gene embeddings remain fixed (not augmented in this setup)

            # --- Forward Pass for Views --- #
            try:
                # Use autocast for mixed precision if enabled
                if args.use_mixed_precision and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        # View 1
                        final_embeddings_view1 = model(raw_omics_view1, gene_embeddings, edge_index_dict_view1)
                        # View 2
                        final_embeddings_view2 = model(raw_omics_view2, gene_embeddings, edge_index_dict_view2)

                        # --- Contrastive Loss (Patients) ---
                        z_patient_view1 = final_embeddings_view1.get('patient')
                        z_patient_view2 = final_embeddings_view2.get('patient')

                        if z_patient_view1 is None or z_patient_view2 is None or z_patient_view1.shape[0] == 0:
                             print(f"Warning: Patient embeddings missing or empty in output at epoch {epoch}. Skipping contrastive loss.")
                             loss_contrastive = torch.tensor(0.0, device=device)
                        else:
                             loss_contrastive = contrastive_loss(z_patient_view1, z_patient_view2, temperature=args.contrastive_temp)

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

                        # --- Omics Reconstruction Loss (use view 1) ---
                        loss_omics = torch.tensor(0.0, device=device)
                        if args.omics_loss_weight > 0 and model.omics_decoder is not None:
                            try:
                                # Decode using embeddings from view 1
                                reconstructed_omics = model.decode_omics(final_embeddings_view1)

                                # Handle different decoder output formats based on decoder type
                                if hasattr(model, 'use_modality_specific_decoders') and model.use_modality_specific_decoders:
                                    # For modality-specific decoders, we get a dictionary of tensors
                                    if isinstance(reconstructed_omics, dict):
                                        # Log shapes for debugging in first epoch
                                        if epoch == 0:
                                            for mod_name, tensor in reconstructed_omics.items():
                                                print(f"Reconstructed {mod_name} shape: {tensor.shape}")

                                        # Calculate loss differently based on mask availability
                                        if gene_masks and not args.ignore_gene_masks:
                                            # With masks: calculate masked loss for each modality separately
                                            total_mod_loss = 0.0
                                            mod_count = 0

                                            # Process each modality separately (except 'concatenated')
                                            for mod_name, rec_mod in reconstructed_omics.items():
                                                if mod_name == 'concatenated':
                                                    continue  # Skip concatenated tensor

                                                # Extract target for this modality
                                                # We extract the appropriate slice from omics_target_tensor
                                                mod_idx = args.decoder_modality_order.index(mod_name) if hasattr(args, 'decoder_modality_order') else mod_count
                                                target_mod = omics_target_tensor[:, :, mod_idx:mod_idx+1]

                                                # Apply modality-specific mask if available
                                                if mod_name in gene_masks:
                                                    # Create tensor mask for this modality
                                                    mask = torch.tensor(gene_masks[mod_name],
                                                                      dtype=rec_mod.dtype,
                                                                      device=rec_mod.device)
                                                    # Reshape mask to [1, num_genes, 1] for broadcasting
                                                    mask = mask.view(1, -1, 1)

                                                    # Calculate masked MSE
                                                    squared_diff = (rec_mod - target_mod) ** 2
                                                    masked_squared_diff = squared_diff * mask
                                                    mask_sum = mask.sum()

                                                    if mask_sum > 0:
                                                        mod_loss = masked_squared_diff.sum() / mask_sum
                                                    else:
                                                        mod_loss = F.mse_loss(rec_mod, target_mod)
                                                else:
                                                    mod_loss = F.mse_loss(rec_mod, target_mod)

                                            total_mod_loss += mod_loss
                                            mod_count += 1

                                        # Average loss across modalities
                                        loss_omics = total_mod_loss / max(1, mod_count)
                                    else:
                                        # Unexpected format - fallback
                                        print(f"Warning: Expected dictionary output from modality-specific decoder, got {type(reconstructed_omics)}")
                                        loss_omics = F.mse_loss(reconstructed_omics, omics_target_tensor)
                                else:
                                    # For standard decoder with concatenated tensor output
                                    # Calculate loss against the original concatenated target
                                    loss_omics = omics_loss_fn(reconstructed_omics, omics_target_tensor)

                                # Additional logging for masked loss
                                if gene_masks and not args.ignore_gene_masks and epoch % 10 == 0:
                                    # We already have masked loss in loss_omics, calculate standard MSE for comparison
                                    if hasattr(model, 'use_modality_specific_decoders') and model.use_modality_specific_decoders:
                                        # For modality-specific decoders, use concatenated tensor
                                        if isinstance(reconstructed_omics, dict) and 'concatenated' in reconstructed_omics:
                                            standard_mse = F.mse_loss(reconstructed_omics['concatenated'], omics_target_tensor)
                                        else:
                                            # Skip comparison if no concatenated tensor available
                                            standard_mse = torch.tensor(float('nan'), device=device)
                                    else:
                                        # For standard decoder
                                        standard_mse = F.mse_loss(reconstructed_omics, omics_target_tensor)

                                    if not torch.isnan(standard_mse):
                                        mask_effect = loss_omics.item() / (standard_mse.item() + 1e-8)
                                        print(f"Epoch {epoch}: Masked MSE: {loss_omics.item():.6f}, "
                                              f"Standard MSE: {standard_mse.item():.6f}, "
                                              f"Mask effect ratio: {mask_effect:.4f}")

                                        # Log to W&B if available
                                        if wandb.run and wandb.run.mode != "disabled":
                                            wandb.log({
                                                "Masking/Standard_MSE": standard_mse.item(),
                                                "Masking/Masked_MSE": loss_omics.item(),
                                                "Masking/Effect_Ratio": mask_effect
                                            })
                            except Exception as decode_err:
                                print(f"Error during omics decoding/loss at epoch {epoch}: {decode_err}")
                                import traceback
                                traceback.print_exc()
                                loss_omics = torch.tensor(0.0, device=device)

                        # --- Structure Preservation Loss ---
                        loss_structure = torch.tensor(0.0, device=device)
                        loss_gene_interaction = torch.tensor(0.0, device=device)

                        if args.unfreeze_gene_embeddings and z_gene_view1 is not None:
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

                        # --- Combine Losses ---
                        # Reconstruction-focused approach
                        if args.training_strategy == 'reconstruction':
                            # Focus on reconstruction losses
                            total_loss = (
                                args.omics_loss_weight * loss_omics +
                                args.graph_loss_weight * loss_graph
                            )

                            # Add structure preservation and gene interaction losses if using unfrozen embeddings
                            if args.unfreeze_gene_embeddings:
                                total_loss += (
                                    args.structure_loss_weight * loss_structure +
                                    args.gene_interaction_loss_weight * loss_gene_interaction
                                )

                            # Optionally add small contrastive loss if desired
                            if args.contrastive_loss_weight > 0:
                                total_loss += args.contrastive_loss_weight * loss_contrastive
                        else:
                            # Original contrastive approach (Strategy 3)
                            total_loss = (
                                loss_contrastive +
                                args.omics_loss_weight * loss_omics
                            )

                        # Print explanation of strategy in first epoch
                        if epoch == 0:
                            if args.training_strategy == 'reconstruction':
                                print(f"STRATEGY 4: Reconstruction-focused training with:")
                                print(f"  - Omics reconstruction loss (weight: {args.omics_loss_weight})")
                                print(f"  - Graph reconstruction loss (weight: {args.graph_loss_weight})")
                                if args.unfreeze_gene_embeddings:
                                    print(f"  - Structure preservation loss (weight: {args.structure_loss_weight})")
                                    print(f"  - Gene interaction loss (weight: {args.gene_interaction_loss_weight})")
                                if args.contrastive_loss_weight > 0:
                                    print(f"  - Small contrastive loss (weight: {args.contrastive_loss_weight})")
                            else:
                                print("STRATEGY 3: Graph reconstruction loss excluded from total_loss since gene embeddings are frozen.")
                                print(f"Training only with: contrastive_loss + {args.omics_loss_weight} * omics_loss")
                else:
                    # Standard precision - same code as before but with the Strategy 3 modifications
                    # View 1
                    final_embeddings_view1 = model(raw_omics_view1, gene_embeddings, edge_index_dict_view1)
                    # View 2
                    final_embeddings_view2 = model(raw_omics_view2, gene_embeddings, edge_index_dict_view2)

                    # --- Contrastive Loss (Patients) ---
                    z_patient_view1 = final_embeddings_view1.get('patient')
                    z_patient_view2 = final_embeddings_view2.get('patient')

                    if z_patient_view1 is None or z_patient_view2 is None or z_patient_view1.shape[0] == 0:
                         print(f"Warning: Patient embeddings missing or empty in output at epoch {epoch}. Skipping contrastive loss.")
                         loss_contrastive = torch.tensor(0.0, device=device)
                    else:
                         loss_contrastive = contrastive_loss(z_patient_view1, z_patient_view2, temperature=args.contrastive_temp)

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

                    # --- Omics Reconstruction Loss (use view 1) ---
                    loss_omics = torch.tensor(0.0, device=device)
                    if args.omics_loss_weight > 0 and model.omics_decoder is not None:
                        try:
                            # Decode using embeddings from view 1
                            reconstructed_omics = model.decode_omics(final_embeddings_view1)

                            # Handle different decoder output formats based on decoder type
                            if hasattr(model, 'use_modality_specific_decoders') and model.use_modality_specific_decoders:
                                # For modality-specific decoders, we get a dictionary of tensors
                                if isinstance(reconstructed_omics, dict):
                                    # Log shapes for debugging in first epoch
                                    if epoch == 0:
                                        for mod_name, tensor in reconstructed_omics.items():
                                            print(f"Reconstructed {mod_name} shape: {tensor.shape}")

                                    # Calculate loss differently based on mask availability
                                    if gene_masks and not args.ignore_gene_masks:
                                        # With masks: calculate masked loss for each modality separately
                                        total_mod_loss = 0.0
                                        mod_count = 0

                                        # Process each modality separately (except 'concatenated')
                                        for mod_name, rec_mod in reconstructed_omics.items():
                                            if mod_name == 'concatenated':
                                                continue  # Skip concatenated tensor

                                            # Extract target for this modality
                                            # We extract the appropriate slice from omics_target_tensor
                                            mod_idx = args.decoder_modality_order.index(mod_name) if hasattr(args, 'decoder_modality_order') else mod_count
                                            target_mod = omics_target_tensor[:, :, mod_idx:mod_idx+1]

                                            # Apply modality-specific mask if available
                                            if mod_name in gene_masks:
                                                # Create tensor mask for this modality
                                                mask = torch.tensor(gene_masks[mod_name],
                                                                  dtype=rec_mod.dtype,
                                                                  device=rec_mod.device)
                                                # Reshape mask to [1, num_genes, 1] for broadcasting
                                                mask = mask.view(1, -1, 1)

                                                # Calculate masked MSE
                                                squared_diff = (rec_mod - target_mod) ** 2
                                                masked_squared_diff = squared_diff * mask
                                                mask_sum = mask.sum()

                                                if mask_sum > 0:
                                                    mod_loss = masked_squared_diff.sum() / mask_sum
                                                else:
                                                    mod_loss = F.mse_loss(rec_mod, target_mod)
                                            else:
                                                mod_loss = F.mse_loss(rec_mod, target_mod)

                                        total_mod_loss += mod_loss
                                        mod_count += 1

                                    # Average loss across modalities
                                    loss_omics = total_mod_loss / max(1, mod_count)
                                else:
                                    # Unexpected format - fallback
                                    print(f"Warning: Expected dictionary output from modality-specific decoder, got {type(reconstructed_omics)}")
                                    loss_omics = F.mse_loss(reconstructed_omics, omics_target_tensor)
                            else:
                                # For standard decoder with concatenated tensor output
                                # Calculate loss against the original concatenated target
                                loss_omics = omics_loss_fn(reconstructed_omics, omics_target_tensor)

                            # Additional logging for masked loss
                            if gene_masks and not args.ignore_gene_masks and epoch % 10 == 0:
                                # We already have masked loss in loss_omics, calculate standard MSE for comparison
                                if hasattr(model, 'use_modality_specific_decoders') and model.use_modality_specific_decoders:
                                    # For modality-specific decoders, use concatenated tensor
                                    if isinstance(reconstructed_omics, dict) and 'concatenated' in reconstructed_omics:
                                        standard_mse = F.mse_loss(reconstructed_omics['concatenated'], omics_target_tensor)
                                    else:
                                        # Skip comparison if no concatenated tensor available
                                        standard_mse = torch.tensor(float('nan'), device=device)
                                else:
                                    # For standard decoder
                                    standard_mse = F.mse_loss(reconstructed_omics, omics_target_tensor)

                                if not torch.isnan(standard_mse):
                                    mask_effect = loss_omics.item() / (standard_mse.item() + 1e-8)
                                    print(f"Epoch {epoch}: Masked MSE: {loss_omics.item():.6f}, "
                                          f"Standard MSE: {standard_mse.item():.6f}, "
                                          f"Mask effect ratio: {mask_effect:.4f}")

                                    # Log to W&B if available
                                    if wandb.run and wandb.run.mode != "disabled":
                                        wandb.log({
                                            "Masking/Standard_MSE": standard_mse.item(),
                                            "Masking/Masked_MSE": loss_omics.item(),
                                            "Masking/Effect_Ratio": mask_effect
                                        })
                        except Exception as decode_err:
                            print(f"Error during omics decoding/loss at epoch {epoch}: {decode_err}")
                            import traceback
                            traceback.print_exc()
                            loss_omics = torch.tensor(0.0, device=device)

                        # --- Structure Preservation Loss ---
                        loss_structure = torch.tensor(0.0, device=device)
                        loss_gene_interaction = torch.tensor(0.0, device=device)

                        if args.unfreeze_gene_embeddings and z_gene_view1 is not None:
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

                        # --- Combine Losses ---
                        # Reconstruction-focused approach
                        if args.training_strategy == 'reconstruction':
                            # Focus on reconstruction losses
                            total_loss = (
                                args.omics_loss_weight * loss_omics +
                                args.graph_loss_weight * loss_graph
                            )

                            # Add structure preservation and gene interaction losses if using unfrozen embeddings
                            if args.unfreeze_gene_embeddings:
                                total_loss += (
                                    args.structure_loss_weight * loss_structure +
                                    args.gene_interaction_loss_weight * loss_gene_interaction
                                )

                            # Optionally add small contrastive loss if desired
                            if args.contrastive_loss_weight > 0:
                                total_loss += args.contrastive_loss_weight * loss_contrastive
                        else:
                            # Original contrastive approach (Strategy 3)
                            total_loss = (
                                loss_contrastive +
                                args.omics_loss_weight * loss_omics
                            )

                        # Print explanation of strategy in first epoch
                        if epoch == 0:
                            if args.training_strategy == 'reconstruction':
                                print(f"STRATEGY 4: Reconstruction-focused training with:")
                                print(f"  - Omics reconstruction loss (weight: {args.omics_loss_weight})")
                                print(f"  - Graph reconstruction loss (weight: {args.graph_loss_weight})")
                                if args.unfreeze_gene_embeddings:
                                    print(f"  - Structure preservation loss (weight: {args.structure_loss_weight})")
                                    print(f"  - Gene interaction loss (weight: {args.gene_interaction_loss_weight})")
                                if args.contrastive_loss_weight > 0:
                                    print(f"  - Small contrastive loss (weight: {args.contrastive_loss_weight})")
                            else:
                                print("STRATEGY 3: Graph reconstruction loss excluded from total_loss since gene embeddings are frozen.")
                                print(f"Training only with: contrastive_loss + {args.omics_loss_weight} * omics_loss")

            except Exception as e:
                 print(f"\nError during forward pass or loss calculation at epoch {epoch}: {e}")
                 print("Skipping backward pass for this epoch.")
                 # Ensure all loss components are scalar Tensors or numbers before logging
                 loss_contrastive = torch.tensor(torch.nan, device=device)
                 loss_graph = torch.tensor(torch.nan, device=device)
                 loss_omics = torch.tensor(torch.nan, device=device)
                 total_loss = torch.tensor(torch.nan, device=device)

            # --- Backward Pass & Optimization --- #
            loss_item = total_loss.item() # Get scalar value for logging/checking
            graph_loss_item = loss_graph.item()
            omics_loss_item = loss_omics.item()
            contrastive_loss_item = loss_contrastive.item()
            structure_loss_item = loss_structure.item() if hasattr(loss_structure, 'item') else 0.0
            gene_interaction_loss_item = loss_gene_interaction.item() if hasattr(loss_gene_interaction, 'item') else 0.0

            if np.isnan(loss_item) or np.isinf(loss_item):
                 print(f"Warning: NaN or Inf loss detected at epoch {epoch}. Skipping backward pass.")
            elif total_loss.requires_grad:
                 if args.use_mixed_precision and device.type == 'cuda' and scaler is not None:
                     scaler.scale(total_loss).backward()
                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     scaler.step(optimizer)
                     scaler.update()
                 else:
                     total_loss.backward()
                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     optimizer.step()
            # No else needed as items are already set

            epoch_duration = time.time() - epoch_start_time
            training_losses.append(loss_item)

            # Update progress bar with appropriate metrics based on training strategy
            if args.training_strategy == 'reconstruction' and args.unfreeze_gene_embeddings:
                train_pbar.set_postfix({
                    'total_loss': f'{loss_item:.4f}',
                    'graph_loss': f'{graph_loss_item:.4f}',
                    'omics_loss': f'{omics_loss_item:.4f}',
                    'struct_loss': f'{structure_loss_item:.4f}',
                    'gene_int_loss': f'{gene_interaction_loss_item:.4f}'
                })
            else:
                train_pbar.set_postfix({
                    'total_loss': f'{loss_item:.4f}',
                    'graph_loss': f'{graph_loss_item:.4f}',
                    'omics_loss': f'{omics_loss_item:.4f}',
                    'contrastive': f'{contrastive_loss_item:.4f}'
                })

            # Log metrics to W&B
            if wandb.run and wandb.run.mode != "disabled":
                log_dict = {
                    "Training/Total_Loss": loss_item,
                    "Training/Contrastive_Loss": contrastive_loss_item,
                    "Training/Graph_Reconstruction_Loss": graph_loss_item,
                    "Training/Omics_Reconstruction_Loss": omics_loss_item,
                    "Training/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Training/Epoch_Duration_sec": epoch_duration,
                    "epoch": epoch
                }

                # Add reconstruction-specific losses if using that strategy
                if args.training_strategy == 'reconstruction' and args.unfreeze_gene_embeddings:
                    log_dict.update({
                        "Training/Structure_Preservation_Loss": structure_loss_item,
                        "Training/Gene_Interaction_Loss": gene_interaction_loss_item,
                        "Training/Gene_Embedding_LR": optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0
                    })

                wandb.log(log_dict)

        print(f"Finished training. Final loss: {training_losses[-1] if training_losses else 'N/A':.4f}")
        # Plot training loss
        valid_losses = [l for l in training_losses if not np.isnan(l) and not np.isinf(l)]
        if valid_losses:
            plt.figure(figsize=(8, 4))
            plt.plot(valid_losses)
            plt.title("Integrated Model Training Loss (Hybrid)")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.grid(True)
            loss_plot_path = os.path.join(args.output_dir, f'integrated_hybrid_training_loss_{args.cancer_type}.png')
            plt.savefig(loss_plot_path)
            print(f"Training loss plot saved to {loss_plot_path}")
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({"Training/Loss_Curve": wandb.Image(loss_plot_path)})
            plt.close()
        else:
            print("No valid training losses recorded to plot.")

    # --- Save Final Model and Embeddings --- #
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n--- Saving Trained Model and Final Embeddings ---")

        # Save Integrated model state dictionary
        model_save_path = os.path.join(args.output_dir, f'integrated_transformer_gcn_{args.cancer_type}.pth')
        try:
             torch.save(model.state_dict(), model_save_path)
             print(f"Integrated model state dict saved to {model_save_path}")
             if wandb.run and wandb.run.mode != "disabled":
                 model_artifact = wandb.Artifact(
                     name=f"integrated-trans-gcn-model-{args.cancer_type}", type="model",
                     description=f"Trained IntegratedTransformerGCN model for {args.cancer_type}"
                 )
                 model_artifact.add_file(model_save_path)
                 wandb.log_artifact(model_artifact)
        except Exception as e:
             print(f"Error saving model state dict: {e}")

        # Generate and save final embeddings (using non-augmented data)
        model.eval()
        with torch.no_grad():
             try:
                 final_embeddings_dict = model(raw_omics_data_dict, gene_embeddings, edge_index_dict)
                 final_patient_embeddings = final_embeddings_dict.get('patient')
                 final_gene_embeddings = final_embeddings_dict.get('gene') # Get final gene embeddings too

                 if final_patient_embeddings is None:
                      print("Error: 'patient' embeddings not found in final model output.")
                 else:
                     final_patient_embeddings = final_patient_embeddings.cpu().numpy()
                     print(f"Final patient embeddings shape: {final_patient_embeddings.shape}")

                     results_data = {
                         'patient_ids': patient_ids,
                         'final_patient_embeddings_gcn': final_patient_embeddings,
                         'gene_list': gene_list, # Final aligned gene list
                         'gene_embeddings': final_gene_embeddings.cpu().numpy() if final_gene_embeddings is not None else None,
                         'training_losses': training_losses,
                         'args': vars(args)
                     }
                     embeddings_save_path = os.path.join(args.output_dir, f'integrated_embeddings_{args.cancer_type}.joblib')
                     joblib.dump(results_data, embeddings_save_path)
                     print(f"Final embeddings saved to {embeddings_save_path}")

                     if wandb.run and wandb.run.mode != "disabled":
                         emb_artifact = wandb.Artifact(
                             name=f"integrated-embeddings-{args.cancer_type}", type="embeddings",
                             description=f"Final patient/gene embeddings from IntegratedTransformerGCN for {args.cancer_type}"
                         )
                         emb_artifact.add_file(embeddings_save_path)
                         wandb.log_artifact(emb_artifact)
             except Exception as e:
                  print(f"Error generating final embeddings: {e}")

    # --- Finalize --- #
    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"\nTotal execution time: {total_execution_time:.2f} seconds")

    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Summary/Total_Execution_Time_sec": total_execution_time,
            "Summary/Final_Loss": training_losses[-1] if training_losses and not np.isnan(training_losses[-1]) else None,
        })
        wandb.finish()

    print("\nIntegrated Training completed successfully!")
    print(f"For clustering and evaluation, use evaluate_risk_stratification_gcn.py (adapted) with embeddings from: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Integrated Transformer+GCN for Patient Stratification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Input Data --- #
    parser.add_argument('--original_data_path', type=str, required=True,
                        help='Path to .joblib file with raw omics_data dict, adj_matrix, and gene_list.')
    parser.add_argument('--gene_embedding_path', type=str, required=False,
                        help='Path to .joblib or .pt file with pre-computed gene_embeddings and gene_list. '
                             'If not provided, defaults to results/autoencoder/joint_ae_embeddings_[cancer_type].joblib')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type key in the original_data file.')

    # --- Patient-Gene Link Construction (using raw omics) --- #
    pg_group = parser.add_argument_group('Patient-Gene Link Construction')
    pg_group.add_argument('--pg_link_omics', type=str, default='rnaseq',
                         help='Raw omics modality key (e.g., rnaseq) to use for patient-gene links.')
    pg_group.add_argument('--pg_link_type', type=str, default='threshold',
                        choices=['threshold', 'top_k_per_patient', 'top_k_per_gene'],
                        help='Method for creating patient-gene edges from the chosen raw omics data.')
    pg_group.add_argument('--pg_link_threshold', type=float, default=0.5,
                       help='Threshold value if pg_link_type is "threshold".')
    pg_group.add_argument('--pg_link_top_k', type=int, default=50,
                       help='Value of K if pg_link_type uses "top_k".')

    # --- Transformer Model --- #
    tf_model_group = parser.add_argument_group('Transformer Model Architecture')
    tf_model_group.add_argument('--transformer_embed_dim', type=int, default=128, help='Core embedding dim for Transformer.')
    tf_model_group.add_argument('--transformer_num_heads', type=int, default=4, help='Number of attention heads.')
    tf_model_group.add_argument('--transformer_ff_dim', type=int, default=256, help='Feed-forward hidden dim in Transformer blocks.')
    tf_model_group.add_argument('--transformer_layers', type=int, default=2, help='Number of Transformer blocks.')
    tf_model_group.add_argument('--transformer_output_dim', type=int, default=64, help='Output dim of Transformer encoder (becomes GCN patient input dim).')
    tf_model_group.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout rate in Transformer.')


    # --- GCN Model (Parameters for the GCN part of the integrated model) --- #
    gcn_model_group = parser.add_argument_group('GCN Model Architecture')
    gcn_model_group.add_argument('--gcn_hidden_dim', type=int, default=128, help='Hidden dimension for GCN layers.')
    gcn_model_group.add_argument('--gcn_output_dim', type=int, default=32, help='Final output dimension for GCN embeddings.')
    gcn_model_group.add_argument('--gcn_layers', type=int, default=2, help='Number of HeteroGCN layers.')
    gcn_model_group.add_argument('--gcn_conv_type', type=str, default='sage', choices=['gcn', 'sage', 'gat'], help='Type of GNN convolution layer.')
    gcn_model_group.add_argument('--gcn_gat_heads', type=int, default=4, help='Number of attention heads if using GATConv.')
    gcn_model_group.add_argument('--gcn_dropout', type=float, default=0.5, help='Dropout rate in GCN layers.')
    gcn_model_group.add_argument('--gcn_no_norm', action='store_true', help='Disable Layer Normalization in GCN layers.')
    gcn_model_group.add_argument('--ignore_gene_masks', action='store_true',
                       help='Ignore gene masks even if they are available in the data.')
    gcn_model_group.add_argument('--use_modality_specific_decoders', action='store_true',
                       help='Use separate decoders for each modality instead of a single concatenated output.')
    gcn_model_group.add_argument('--decoder_activation', type=str, default='sigmoid', choices=['sigmoid', 'relu', 'none'],
                       help='Activation function to use for the omics decoder.')
    gcn_model_group.add_argument('--decoder_patient_batch_size', type=int, default=1,
                       help='Number of patients to process at once during decoding to reduce memory usage. Set to 1 for minimal memory usage, increase if GPU memory allows.')
    gcn_model_group.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing to reduce memory usage at the cost of some increased computation time.')
    gcn_model_group.add_argument('--reduce_decoder_memory', action='store_true',
                       help='Use smaller intermediate dimensions in the decoder to reduce memory usage.')
    gcn_model_group.add_argument('--use_mixed_precision', action='store_true',
                       help='Enable automatic mixed precision training to reduce memory usage.')
    gcn_model_group.add_argument('--cuda_alloc_conf', type=str, default='max_split_size_mb:128',
                       help='Configuration string for PYTORCH_CUDA_ALLOC_CONF environment variable to control CUDA memory allocation behavior.')

    # --- Training --- #
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--train_epochs', type=int, default=100, help='Number of epochs to train the integrated model.')
    train_group.add_argument('--gcn_lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    train_group.add_argument('--gcn_weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer.')
    train_group.add_argument('--contrastive_temp', type=float, default=0.1, help='Temperature for contrastive loss.')
    train_group.add_argument('--graph_loss_weight', type=float, default=0.4, help='Weight for the graph reconstruction loss term.')
    train_group.add_argument('--omics_loss_weight', type=float, default=0.2, help='Weight for the omics reconstruction loss term.')
    train_group.add_argument('--aug_feature_mask_rate', type=float, default=0.3, help='Probability of masking raw omics features during augmentation.')
    train_group.add_argument('--aug_edge_drop_rate', type=float, default=0.2, help='Probability of dropping graph edges during augmentation.')

    # --- Reconstruction-Focused Training Parameters --- #
    recon_group = parser.add_argument_group('Reconstruction-Focused Training')
    recon_group.add_argument('--training_strategy', type=str, default='contrastive',
                          choices=['contrastive', 'reconstruction'],
                          help='Training strategy: contrastive (original) or reconstruction (new approach).')
    recon_group.add_argument('--unfreeze_gene_embeddings', action='store_true',
                          help='Allow gene embeddings to be updated during training with controlled learning.')
    recon_group.add_argument('--gene_lr', type=float, default=0.0001,
                          help='Learning rate for gene embeddings (10x smaller than main lr by default).')
    recon_group.add_argument('--contrastive_loss_weight', type=float, default=0.0,
                          help='Weight for contrastive loss when using reconstruction strategy (0 to disable).')
    recon_group.add_argument('--structure_loss_weight', type=float, default=0.5,
                          help='Weight for structure preservation loss to maintain original embedding structure.')
    recon_group.add_argument('--gene_interaction_loss_weight', type=float, default=0.3,
                          help='Weight for gene-gene interaction loss to refine embeddings based on interactions.')
    recon_group.add_argument('--gene_interaction_pos_weight', type=float, default=1.0,
                          help='Weight for positive interactions in gene interaction loss.')
    recon_group.add_argument('--gene_interaction_neg_weight', type=float, default=0.1,
                          help='Weight for negative interactions (non-edges) in gene interaction loss.')

    # --- Output & Misc --- #
    output_group = parser.add_argument_group('Output & Miscellaneous')
    output_group.add_argument('--output_dir', type=str, default='./integrated_training_results',
                        help='Directory to save results (model, embeddings).')


    args = parser.parse_args()

    # Set default gene embedding path based on cancer type if not provided
    if args.gene_embedding_path is None:
        args.gene_embedding_path = f"results/autoencoder/joint_ae_embeddings_{args.cancer_type}.joblib"
        print(f"Using default gene embedding path: {args.gene_embedding_path}")

    # --- Create unique output directory --- #
    if args.output_dir:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add encoder/decoder info to directory name
        decoder_info = "mod_dec" if args.use_modality_specific_decoders else "cat_dec"
        strategy_info = f"recon_{args.structure_loss_weight}s_{args.gene_interaction_loss_weight}g" if args.training_strategy == 'reconstruction' else "contrast"
        gene_emb_info = "unfrozen" if args.unfreeze_gene_embeddings else "frozen"
        run_folder_name = f"{args.cancer_type}_integrated_T{args.transformer_layers}L_G{args.gcn_layers}L_{args.gcn_conv_type}_{decoder_info}_{strategy_info}_{gene_emb_info}_E{args.train_epochs}_{time_stamp}"
        args.output_dir = os.path.join(args.output_dir, run_folder_name)
        try:
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Created output directory: {args.output_dir}")
        except OSError as e:
             print(f"Error creating output directory {args.output_dir}: {e}")
             args.output_dir = f"./integrated_training_results/run_{time_stamp}" # Fallback
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Using fallback output directory: {args.output_dir}")

    # --- Run Training --- #
    run_integrated_training(args)

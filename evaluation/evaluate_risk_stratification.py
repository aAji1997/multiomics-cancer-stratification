import torch
import numpy as np
import pandas as pd
import joblib
import os
import time
import argparse
import json
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
import scipy.sparse as sp

import hdbscan

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from modelling.gcn.hetero_gcn_model import HeteroGCN
from modelling.autoencoder.model import JointAutoencoder
from modelling.gcn.integrated_transformer_gcn_model import IntegratedTransformerGCN

from modelling.gcn.train_risk_stratification_gcn import load_omics_for_links, create_patient_gene_edges

import wandb

# Load WandB API key from file
try:
    with open('.api_config.json', 'r') as f:
        config = json.load(f)
        WANDB_API_KEY = config['wandb_api_key']
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
except Exception as e:
    print(f"Warning: Error loading W&B API key: {e}. W&B tracking may not work properly.")

def run_survival_analysis(patient_ids, clusters, clinical_df, output_dir, cancer_type):
    """Performs survival analysis using lifelines."""
    if clinical_df is None:
        print("Skipping survival analysis: Clinical data not provided or failed to load.")
        return

    print("\n--- Performing Survival Analysis ---")
    # Create DataFrame for analysis
    cluster_df = pd.DataFrame({'patient_id': patient_ids, 'cluster': clusters})
    
    # Filter out noise points (cluster = -1) from HDBSCAN
    if -1 in cluster_df['cluster'].values:
        noise_count = (cluster_df['cluster'] == -1).sum()
        print(f"Filtering out {noise_count} noise points (cluster = -1) from survival analysis")
        cluster_df = cluster_df[cluster_df['cluster'] != -1]
        
        if cluster_df.empty:
            print("Error: All patients were classified as noise. Cannot perform survival analysis.")
            return
    
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

    # Check if the required columns are present, if not, try to map from alternative column names
    if not all(col in analysis_df.columns for col in required_cols):
        print(f"Required columns not found. Checking for alternative column names...")
        
        # Map 'overall_survival' to 'duration' if available
        if 'duration' not in analysis_df.columns and 'overall_survival' in analysis_df.columns:
            analysis_df['duration'] = analysis_df['overall_survival']
            print("Mapped 'overall_survival' to 'duration'")
        
        # Map 'status' to 'event' if available
        if 'event' not in analysis_df.columns and 'status' in analysis_df.columns:
            analysis_df['event'] = analysis_df['status']
            print("Mapped 'status' to 'event'")
            
        # Check if we now have the required columns
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
            
            # Log p-value to wandb
            if wandb.run and wandb.run.mode != "disabled":
                wandb.log({"Survival/Logrank_P_Value": p_value})
            
        except Exception as e:
            print(f"Error during log-rank test: {e}")
    else:
         print("Skipping log-rank test (only one cluster).")

    # Kaplan-Meier plots
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    # Create a table of cluster statistics for wandb
    cluster_stats = []

    for cluster_label in sorted(analysis_df['cluster'].unique()):
        # Skip noise points in the visualization and statistics
        if cluster_label == -1:
            continue
            
        subset = analysis_df[analysis_df['cluster'] == cluster_label]
        kmf.fit(subset['duration'], event_observed=subset['event'], label=f'Cluster {cluster_label} (n={len(subset)})')
        kmf.plot_survival_function(ax=ax)
        
        # Record stats for wandb
        events = subset['event'].sum()
        median_survival = kmf.median_survival_time_ if hasattr(kmf, 'median_survival_time_') else float('nan')
        cluster_stats.append({
            "cluster": str(int(cluster_label)),
            "size": len(subset),
            "events": int(events),
            "censored": len(subset) - int(events),
            "median_survival_days": float(median_survival) if not np.isnan(median_survival) else None
        })

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
    
    # Save plot to wandb
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({"Survival/Kaplan_Meier_Plot": wandb.Image(plt)})
        
        # Log cluster statistics as a table
        wandb.log({"Survival/Cluster_Stats": wandb.Table(
            columns=["cluster", "size", "events", "censored", "median_survival_days"],
            data=[[s["cluster"], s["size"], s["events"], s["censored"], s["median_survival_days"]] for s in cluster_stats]
        )})
    
    # Still save the plot locally for reference
    plot_save_path = os.path.join(output_dir, f'kaplan_meier_{cancer_type}.png')
    plt.savefig(plot_save_path)
    print(f"Kaplan-Meier plot saved to {plot_save_path}")
    plt.close()

def run_evaluation(args):
    """
    Perform inference, clustering, and visualization using a trained GCN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()
    
    # --- Initialize Weights & Biases ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}-eval-{args.cancer_type}-{run_timestamp}"
    project_name = f"cancer-stratification-eval" # Fixed project name for all evaluations
    
    print("Initializing Weights & Biases...")
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=vars(args)
        )
        print(f"W&B Run URL: {wandb.run.get_url()}")
    except Exception as e:
        print(f"Error initializing W&B: {e}. Proceeding without W&B tracking.")
        os.environ["WANDB_DISABLED"] = "true"
        wandb.init(mode="disabled")
    
    # --- Determine paths based on model_type --- 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, 'results')
    
    # Set default paths based on model type if not explicitly provided
    if args.embedding_path is None and args.model_path is None:
        if args.model_type == 'autoencoder':
            model_dir = os.path.join(results_dir, 'autoencoder')
            default_embedding_path = os.path.join(model_dir, f'joint_ae_embeddings_{args.cancer_type}.joblib')
            if os.path.exists(default_embedding_path):
                args.embedding_path = default_embedding_path
                print(f"Using default autoencoder embeddings path: {args.embedding_path}")
            else:
                print(f"Default autoencoder embeddings not found at: {default_embedding_path}")
                
        elif args.model_type == 'gcn':
            model_dir = os.path.join(results_dir, 'gcn')
            # Look for most recent GCN embedding files
            pattern = f'gcn_embeddings_{args.cancer_type}_*.joblib'
            import glob
            gcn_files = sorted(glob.glob(os.path.join(model_dir, pattern)), reverse=True)
            if gcn_files:
                args.embedding_path = gcn_files[0]
                print(f"Using most recent GCN embeddings: {args.embedding_path}")
            else:
                print(f"No GCN embedding files found matching pattern: {pattern}")
                
        elif args.model_type == 'integrated':
            model_dir = os.path.join(results_dir, 'integrated')
            default_embedding_path = os.path.join(model_dir, f'integrated_embeddings_{args.cancer_type}.joblib')
            if os.path.exists(default_embedding_path):
                args.embedding_path = default_embedding_path
                print(f"Using default integrated embeddings path: {args.embedding_path}")
            else:
                print(f"Default integrated embeddings not found at: {default_embedding_path}")
    
    # Log selected paths to wandb as config values (not as logged metrics)
    if wandb.run and wandb.run.mode != "disabled":
        # Use config instead of log for these path values
        wandb.config.update({
            "Embedding_Path": args.embedding_path,
            "Model_Path": args.model_path,
            "Original_Data_Path": args.original_data_path,
            "Clinical_Data_Path": args.clinical_data_path
        })

    # --- Load Data --- #
    print(f"\nLoading data for evaluation")
    
    patient_ids = None
    gene_list = None
    patient_embeddings = None
    gene_embeddings = None
    final_patient_embeddings_gcn = None
    
    # Load pre-computed embeddings or set up for model inference
    if args.model_path and os.path.exists(args.model_path):
        # We'll load the trained model and generate embeddings from it
        print(f"Loading trained model from: {args.model_path}")
        model_exists = True
    elif args.embedding_path and os.path.exists(args.embedding_path):
        # We'll use pre-computed embeddings directly
        print(f"Loading pre-computed embeddings from: {args.embedding_path}")
        model_exists = False
        try:
            embeddings_data = joblib.load(args.embedding_path)
            
            # Check what kind of embeddings file we have
            if args.model_type == 'autoencoder':
                # Expected keys for autoencoder: gene_embeddings, patient_embeddings, patient_ids, gene_list
                if all(k in embeddings_data for k in ['gene_embeddings', 'patient_embeddings', 'patient_ids', 'gene_list']):
                    gene_embeddings = torch.tensor(embeddings_data['gene_embeddings'], dtype=torch.float32)
                    patient_embeddings = torch.tensor(embeddings_data['patient_embeddings'], dtype=torch.float32)
                    patient_ids = list(np.array(embeddings_data['patient_ids'])) 
                    gene_list = list(np.array(embeddings_data['gene_list']))
                    # For autoencoder, we'll use patient_embeddings directly for clustering
                    final_patient_embeddings_gcn = embeddings_data['patient_embeddings']
                    print(f"Loaded autoencoder embeddings: {len(patient_ids)} patients and {len(gene_list)} genes.")
                    tsne_embeddings = None
                else:
                    print(f"Error: Autoencoder embeddings file missing expected keys.")
                    return
            elif 'cluster_assignments' in embeddings_data:  # This is a GCN/integrated results file
                print("Found results file with cluster assignments.")
                patient_ids = list(np.array(embeddings_data['patient_ids']))
                gene_list = list(np.array(embeddings_data['gene_list']))
                final_patient_embeddings_gcn = embeddings_data.get('final_patient_embeddings_gcn', 
                                                                embeddings_data.get('final_patient_embeddings'))
                
                # If we already have cluster assignments, we can skip clustering
                if args.use_existing_clusters and 'cluster_assignments' in embeddings_data:
                    patient_clusters = embeddings_data['cluster_assignments']
                    print(f"Using {len(np.unique(patient_clusters))} existing clusters from file.")
                
                # If TSNE embeddings are already computed, we can use them
                if 'tsne_embeddings' in embeddings_data and embeddings_data['tsne_embeddings'] is not None:
                    tsne_embeddings = embeddings_data['tsne_embeddings']
                    print(f"Using pre-computed TSNE embeddings from file.")
                else:
                    tsne_embeddings = None
            else:  # This is just a raw embeddings file 
                # Check for GCN and Integrated model embeddings format first (different key names)
                if 'final_patient_embeddings_gcn' in embeddings_data:
                    # Handle GCN or Integrated model embeddings
                    patient_ids = list(np.array(embeddings_data['patient_ids'])) 
                    gene_list = list(np.array(embeddings_data['gene_list']))
                    final_patient_embeddings_gcn = embeddings_data['final_patient_embeddings_gcn']
                    
                    # Check for gene embeddings (could be named differently or missing)
                    if 'gene_embeddings' in embeddings_data and embeddings_data['gene_embeddings'] is not None:
                        gene_embeddings = torch.tensor(embeddings_data['gene_embeddings'], dtype=torch.float32)
                        print(f"Loaded embeddings with key 'gene_embeddings': {len(patient_ids)} patients and {len(gene_list)} genes.")
                    else:
                        # Older GCN format without gene embeddings or different key name
                        gene_embeddings = None
                        print(f"Loaded embeddings with key 'final_patient_embeddings_gcn': {len(patient_ids)} patients. (Gene embeddings not available)")
                    
                    # Patient embeddings will be accessed through final_patient_embeddings_gcn
                    patient_embeddings = None
                    tsne_embeddings = None
                    
                # Also try alternative key names used by the integrated model
                elif 'final_patient_embeddings' in embeddings_data:
                    # Handle legacy integrated model format
                    patient_ids = list(np.array(embeddings_data['patient_ids'])) 
                    gene_list = list(np.array(embeddings_data['gene_list']))
                    final_patient_embeddings_gcn = embeddings_data['final_patient_embeddings']  # Assign to expected variable
                    
                    # Check for gene embeddings (could be named 'final_gene_embeddings' in older files)
                    if 'final_gene_embeddings' in embeddings_data and embeddings_data['final_gene_embeddings'] is not None:
                        gene_embeddings = torch.tensor(embeddings_data['final_gene_embeddings'], dtype=torch.float32)
                        print(f"Loaded embeddings with key 'final_gene_embeddings': {len(patient_ids)} patients and {len(gene_list)} genes.")
                    else:
                        gene_embeddings = None
                        print(f"Loaded embeddings with key 'final_patient_embeddings': {len(patient_ids)} patients. (Gene embeddings not available)")
                    
                    patient_embeddings = None
                    tsne_embeddings = None
                else:
                    # Standard raw embeddings format (e.g., autoencoder)
                    gene_embeddings = torch.tensor(embeddings_data['gene_embeddings'], dtype=torch.float32)
                    patient_embeddings = torch.tensor(embeddings_data['patient_embeddings'], dtype=torch.float32)
                    patient_ids = list(np.array(embeddings_data['patient_ids'])) 
                    gene_list = list(np.array(embeddings_data['gene_list']))
                    print(f"Loaded standard format embeddings: {len(patient_ids)} patients and {len(gene_list)} genes.")
                    # We'll need to run these through the model
                    final_patient_embeddings_gcn = None
                    tsne_embeddings = None
        except FileNotFoundError:
            print(f"Error: Embedding file not found at {args.embedding_path}")
            return
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return
    else:
        print("Error: Either model_path or embedding_path must be provided and exist")
        return
        
    # Log basic embedding info to wandb
    if wandb.run and wandb.run.mode != "disabled" and patient_ids is not None:
        # Use numeric values with log, but put string value in config
        wandb.log({
            "Data/Num_Patients": len(patient_ids) if patient_ids else 0,
            "Data/Num_Genes": len(gene_list) if gene_list else 0
        })
        wandb.config.update({"Model_Type": args.model_type})

    # --- Load Clinical Data if Provided --- #
    clinical_df = None
    if args.clinical_data_path:
         print(f"\nLoading clinical data from: {args.clinical_data_path}")
         try:
             # Assuming CSV/TSV format, adjust as needed
             sep = '\t' if args.clinical_data_path.endswith('.tsv') else ','
             clinical_df = pd.read_csv(args.clinical_data_path, sep=sep)
             print(f"Loaded clinical data for {len(clinical_df)} records initially.")
         except FileNotFoundError:
             print(f"Error: Clinical data file not found at {args.clinical_data_path}")
             clinical_df = None # Ensure it's None if loading fails
         except Exception as e:
             print(f"Error loading or processing clinical data: {e}")
             clinical_df = None # Ensure it's None if loading fails
    else:
         print("\nNo clinical data path provided. Survival analysis will be skipped.")

    # --- Construct HeteroData for Model Inference (if needed) --- #
    if model_exists or final_patient_embeddings_gcn is None:
        # We need to run the model inference or recreate the graph for model inference
        print("\nLoading original data and constructing heterogeneous graph...")
        
        adj_matrix_gene_gene = None
        omics_df_for_links = None
        
        # We need the original data to construct the graph
        if not args.original_data_path:
            print("Error: Original data path must be provided to construct the graph for model inference")
            return
            
        try:
            prepared_data = joblib.load(args.original_data_path)
            if args.cancer_type not in prepared_data:
                 raise KeyError(f"Cancer type '{args.cancer_type}' not found in {args.original_data_path}.")
            cancer_data = prepared_data[args.cancer_type]
            
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
                print("Attempting to proceed assuming adj_matrix rows/cols correspond to embedding gene_list order, but results may be incorrect.")
                # Check dimensions as a basic safeguard
                if adj_matrix_gene_gene.shape[0] != len(gene_list):
                     print(f"Error: Adjacency matrix dimension ({adj_matrix_gene_gene.shape[0]}) mismatch with embedding gene list ({len(gene_list)}). Cannot proceed.")
                     return
            
            # Load specific omics for patient-gene links
            omics_df_for_links = load_omics_for_links(args.original_data_path, args.cancer_type, args.pg_link_omics)
            if omics_df_for_links is None:
                 print("Warning: Failed to load omics data for patient-gene links. Links will not be created.")

        except FileNotFoundError:
            print(f"Error: Original data file not found at {args.original_data_path}")
            return
        except KeyError as e:
             print(f"Error accessing data in original file: {e}")
             return
        except Exception as e:
            print(f"Error loading original data: {e}")
            return
            
        # Construct HeteroData Object
        hetero_data = HeteroData()

        # Node Features
        hetero_data['patient'].x = patient_embeddings
        # Handle case where gene_embeddings might be None
        if gene_embeddings is None:
            print("Gene embeddings not available in loaded file. Using identity features.")
            # Create identity features for genes (one-hot encoding)
            num_genes = len(gene_list)
            # For memory efficiency, use a sparse identity matrix if the number of genes is large
            if num_genes > 10000:
                identity_features = sp.eye(num_genes).tocoo()
                row, col = identity_features.row, identity_features.col
                indices = torch.tensor(np.vstack((row, col)), dtype=torch.long)
                values = torch.tensor(identity_features.data, dtype=torch.float32)
                shape = identity_features.shape
                gene_features = torch.sparse.FloatTensor(indices, values, shape).to_dense()
            else:
                gene_features = torch.eye(num_genes, dtype=torch.float32)
            hetero_data['gene'].x = gene_features
            print(f"Created identity features for {num_genes} genes.")
        else:
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

        # Patient-Gene Edges (Require omics_df_for_links to be loaded)
        if omics_df_for_links is not None:
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
             print("Skipping patient-gene edges as omics data was not loaded.")
             hetero_data['patient', 'expresses', 'gene'].edge_index = torch.empty((2,0), dtype=torch.long)
             hetero_data['gene', 'rev_expresses', 'patient'].edge_index = torch.empty((2,0), dtype=torch.long)

        # Move data to device
        try:
            hetero_data = hetero_data.to(device)
            print(f"Moved HeteroData to {device}.")
        except Exception as e:
            print(f"Error moving HeteroData to device {device}: {e}")
            return

        print("\nHeteroData structure:")
        print(hetero_data)
        
        # Load the model if we have a model path
        if model_exists:
            # Special case for autoencoder: need to load raw omics data and run inference directly
            if args.model_type == 'autoencoder':
                print("\n--- Loading JointAutoencoder model and raw omics data for inference ---")
                try:
                    # Load the prepared data to get the raw omics data
                    if not args.original_data_path:
                        print("Error: original_data_path is required when using model_path with autoencoder")
                        return
                    
                    prepared_data = joblib.load(args.original_data_path)
                    if args.cancer_type not in prepared_data:
                        raise KeyError(f"Cancer type '{args.cancer_type}' not found in {args.original_data_path}")
                        
                    cancer_data = prepared_data[args.cancer_type]
                    if 'omics_data' not in cancer_data or not cancer_data['omics_data']:
                        raise KeyError(f"'omics_data' not found or empty for {args.cancer_type}")
                    
                    # Get omics_data dict
                    omics_data_dict = cancer_data['omics_data']
                    gene_list = list(np.array(cancer_data['gene_list']))
                    
                    # Create JointOmicsDataset (we'll need to import this)
                    from modelling.autoencoder.data_utils import JointOmicsDataset
                    
                    # Use all available modalities or the ones specified
                    modalities_to_use = list(omics_data_dict.keys())
                    # Remove clinical if present (not needed for the model)
                    if 'clinical' in modalities_to_use:
                        modalities_to_use.remove('clinical')
                    
                    print(f"Using omics modalities: {modalities_to_use}")
                    
                    # Create dataset
                    joint_dataset = JointOmicsDataset(omics_data_dict, gene_list, modalities=modalities_to_use)
                    
                    # Get input dimensions for model instantiation
                    omics_dims = {mod: joint_dataset.omics_data[mod].shape[1] for mod in modalities_to_use}
                    
                    # We would need to know the model architecture parameters that were used during training
                    # Attempt to use defaults that are commonly used (imperfect but might work)
                    model = JointAutoencoder(
                        num_nodes=len(gene_list),
                        modality_latent_dims={mod: min(128, dim) for mod, dim in omics_dims.items()},
                        modality_order=modalities_to_use,
                        graph_feature_dim=len(gene_list),  # Using identity features
                        gene_embedding_dim=args.gene_embedding_dim or 64,
                        patient_embedding_dim=args.patient_embedding_dim or 128,
                        graph_dropout=args.gcn_dropout
                    ).to(device)
                    
                    # Load model weights
                    model.load_state_dict(torch.load(args.model_path, map_location=device))
                    print(f"Successfully loaded JointAutoencoder model from {args.model_path}")
                    
                    # Run inference
                    model.eval()
                    
                    # Create a DataLoader for batch processing
                    from torch.utils.data import DataLoader
                    dataloader = DataLoader(joint_dataset, batch_size=args.batch_size or 16, shuffle=False)
                    
                    # Track patient IDs and collect embeddings
                    patient_ids = joint_dataset.patient_ids
                    all_patient_embeddings = []
                    
                    # Prepare graph data
                    from modelling.autoencoder.data_utils import prepare_graph_data
                    # Use identity matrix or other graph init method
                    graph_node_features, graph_edge_index, graph_edge_weight, _ = prepare_graph_data(
                        adj_matrix_gene_gene, 
                        node_init_modality='identity'
                    )
                    
                    # Move to device
                    graph_node_features = graph_node_features.to(device)
                    graph_edge_index = graph_edge_index.to(device)
                    graph_edge_weight = graph_edge_weight.to(device)
                    
                    with torch.no_grad():
                        # Get gene embeddings from model (this will be mu from VGAE during eval mode)
                        mu_final, _, _ = model.graph_autoencoder.encode(
                            graph_node_features, graph_edge_index, edge_weight=graph_edge_weight
                        )
                        gene_embeddings = mu_final.cpu().numpy()
                        
                        # Process each batch to get patient embeddings
                        for omics_batch in tqdm(dataloader, desc="Processing batches"):
                            omics_batch = omics_batch.to(device)
                            # Using z_gene (mu) during inference
                            z_gene_eval = mu_final  # deterministic embedding
                            # Get patient embeddings
                            patient_batch_embeddings = model.omics_processor.encode(omics_batch, z_gene_eval)
                            all_patient_embeddings.append(patient_batch_embeddings.cpu().numpy())
                    
                    # Concatenate all batches
                    final_patient_embeddings_gcn = np.concatenate(all_patient_embeddings, axis=0)
                    print(f"Generated patient embeddings with shape: {final_patient_embeddings_gcn.shape}")
                    
                except Exception as e:
                    print(f"Error during JointAutoencoder inference: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                # For GCN and Integrated models
                metadata = hetero_data.metadata()
                node_feature_dims = {ntype: hetero_data[ntype].x.shape[1] for ntype in metadata[0]}
                
                # Create model instance with the same architecture based on model_type
                try:
                    if args.model_type == 'gcn':
                        # Standard HeteroGCN
                        model = HeteroGCN(
                            metadata=metadata,
                            node_feature_dims=node_feature_dims,
                            hidden_channels=args.gcn_hidden_dim,
                            out_channels=args.gcn_output_dim, 
                            num_layers=args.gcn_layers,
                            conv_type=args.gcn_conv_type,
                            num_heads=args.gcn_gat_heads,
                            dropout_rate=args.gcn_dropout,
                            use_layer_norm=not args.gcn_no_norm
                        ).to(device)
                    elif args.model_type == 'integrated':
                        # IntegratedTransformerGCN (we need raw omics data too, but for inference we can skip that part)
                        model = IntegratedTransformerGCN(
                            # We need these for loading the model but don't need actual values for inference
                            # as we'll load pre-trained weights
                            omics_input_dims={},
                            transformer_embed_dim=64,
                            transformer_num_heads=4,
                            transformer_ff_dim=128,
                            num_transformer_layers=2,
                            transformer_output_dim=64,
                            transformer_dropout=0.1,
                            gcn_metadata=metadata,
                            gene_feature_dim=hetero_data['gene'].x.shape[1],
                            gcn_hidden_channels=args.gcn_hidden_dim,
                            gcn_out_channels=args.gcn_output_dim,
                            gcn_num_layers=args.gcn_layers,
                            gcn_conv_type=args.gcn_conv_type,
                            gcn_num_heads=args.gcn_gat_heads,
                            gcn_dropout_rate=args.gcn_dropout,
                            gcn_use_layer_norm=not args.gcn_no_norm
                        ).to(device)
                    
                    # Load saved weights
                    model.load_state_dict(torch.load(args.model_path, map_location=device))
                    print(f"Successfully loaded model from {args.model_path}")
                    
                    # Run inference
                    model.eval()
                    with torch.no_grad():
                        if args.model_type == 'integrated':
                            # For integrated model, we need special handling as it has a different forward signature
                            # but since we're using pre-computed embeddings, we can just use the GCN part
                            final_embeddings_dict = model.forward_gcn_only(hetero_data.x_dict, hetero_data.edge_index_dict)
                        else:
                            final_embeddings_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
                        
                    if 'patient' not in final_embeddings_dict:
                        print("Error: 'patient' embeddings not found in model output.")
                        return
                    
                    final_patient_embeddings_gcn = final_embeddings_dict['patient'].cpu().numpy()
                    print(f"Generated patient embeddings with shape: {final_patient_embeddings_gcn.shape}")
                    
                except Exception as e:
                    print(f"Error loading or running model: {e}")
                    return

    # --- Dimensionality Reduction with TSNE (if not already done) --- #
    if tsne_embeddings is None and final_patient_embeddings_gcn is not None:
        if final_patient_embeddings_gcn.shape[0] > 1:  # Need at least 2 samples for TSNE
            try:
                print("\n--- Performing TSNE dimensionality reduction ---")
                tsne = TSNE(n_components=args.tsne_components, 
                            perplexity=min(args.tsne_perplexity, final_patient_embeddings_gcn.shape[0]-1), 
                            random_state=42, n_jobs=-1)  # Use all available cores
                tsne_embeddings = tsne.fit_transform(final_patient_embeddings_gcn)
                print(f"TSNE embeddings shape: {tsne_embeddings.shape}")
                
                # Visualize TSNE embeddings as a scatter plot if 2D
                if args.tsne_components == 2:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0.7)
                    plt.title(f"TSNE visualization of patient embeddings ({args.cancer_type})")
                    plt.tight_layout()
                    
                    # Log to wandb
                    if wandb.run and wandb.run.mode != "disabled":
                        wandb.log({"Visualization/TSNE_Embeddings": wandb.Image(plt)})
                    
                    # Save locally as well
                    tsne_plot_path = os.path.join(args.output_dir, f'tsne_visualization_{args.cancer_type}.png')
                    plt.savefig(tsne_plot_path)
                    plt.close()
                    print(f"TSNE visualization saved to {tsne_plot_path}")
            except Exception as e:
                print(f"Error during TSNE dimensionality reduction: {e}")
                tsne_embeddings = None
        else:
            print("Skipping TSNE (insufficient number of patients).")

    # --- Perform Clustering (if embeddings available and no existing clusters) --- #
    patient_clusters = None
    silhouette_avg = None
    
    # Check if we already have clusters loaded from file
    if 'patient_clusters' in locals() and patient_clusters is not None:
        print(f"Using existing clusters from loaded file")
    elif final_patient_embeddings_gcn is not None:
        # Choose embeddings to cluster (TSNE or original)
        embeddings_to_cluster = tsne_embeddings if tsne_embeddings is not None else final_patient_embeddings_gcn
        
        if embeddings_to_cluster.shape[0] < 1:
            print("Error: No patient embeddings available for clustering.")
        else:
            try:
                if args.use_hdbscan:
                    print(f"\n--- Performing HDBSCAN clustering ---")
                    # Set min_cluster_size based on dataset size or argument
                    auto_min_cluster_size = max(5, int(embeddings_to_cluster.shape[0] * 0.05))  # At least 5 or 5% of data
                    min_cluster_size = args.hdbscan_min_cluster_size if args.hdbscan_min_cluster_size > 0 else auto_min_cluster_size
                    
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=args.hdbscan_min_samples,
                        cluster_selection_epsilon=args.hdbscan_epsilon,
                        prediction_data=True
                    )
                    patient_clusters = clusterer.fit_predict(embeddings_to_cluster)
                    
                    # Count number of clusters (excluding noise points marked as -1)
                    unique_clusters = np.unique(patient_clusters)
                    num_clusters = len(unique_clusters[unique_clusters >= 0])
                    noise_points = np.sum(patient_clusters == -1)
                    
                    print(f"HDBSCAN found {num_clusters} clusters with {noise_points} noise points.")
                    print(f"Cluster distribution: {np.bincount(patient_clusters[patient_clusters >= 0])}")
                    
                    # If all patients are classified as noise, fall back to KMeans
                    if num_clusters == 0:
                        print("Warning: HDBSCAN classified all points as noise. Falling back to KMeans.")
                        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init='auto')
                        patient_clusters = kmeans.fit_predict(embeddings_to_cluster)
                else:
                    # Fall back to KMeans
                    method = "KMeans" if args.use_hdbscan else "KMeans (as requested)"
                    print(f"\n--- Performing {method} clustering with K={args.num_clusters} ---")
                    
                    if embeddings_to_cluster.shape[0] < args.num_clusters:
                        print(f"Warning: Number of patients ({embeddings_to_cluster.shape[0]}) is less than K ({args.num_clusters}). Setting K to number of patients.")
                        num_clusters = embeddings_to_cluster.shape[0]
                    else:
                        num_clusters = args.num_clusters
                    
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                    patient_clusters = kmeans.fit_predict(embeddings_to_cluster)
                
                # Evaluate clustering if there's more than one cluster
                unique_clusters = np.unique(patient_clusters)
                if len(unique_clusters) > 1 and embeddings_to_cluster.shape[0] > 1:
                    # Filter out noise points for silhouette calculation
                    if -1 in unique_clusters:
                        valid_indices = patient_clusters != -1
                        if np.sum(valid_indices) > 1:  # Need at least 2 points
                            valid_embeddings = embeddings_to_cluster[valid_indices]
                            valid_clusters = patient_clusters[valid_indices]
                            silhouette_avg = silhouette_score(valid_embeddings, valid_clusters)
                            print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
                    else:
                        silhouette_avg = silhouette_score(embeddings_to_cluster, patient_clusters)
                        print(f"Silhouette Score: {silhouette_avg:.4f}")
                        
                # Visualize clustering results if using TSNE with 2 components
                if tsne_embeddings is not None and args.tsne_components == 2:
                    plt.figure(figsize=(10, 8))
                    
                    # Plot with different colors for each cluster
                    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=patient_clusters, 
                                         cmap='viridis', alpha=0.7)
                    plt.title(f"Clustering results on TSNE embeddings ({args.cancer_type})")
                    
                    # Add legend showing cluster labels
                    handles, labels = scatter.legend_elements()
                    legend_labels = [f"Cluster {i}" if i != -1 else "Noise" for i in range(-1, len(np.unique(patient_clusters))-1)]
                    plt.legend(handles, legend_labels, title="Clusters")
                    
                    plt.tight_layout()
                    
                    # Log to wandb
                    if wandb.run and wandb.run.mode != "disabled":
                        wandb.log({
                            "Clustering/Visualization": wandb.Image(plt),
                            "Clustering/Silhouette_Score": silhouette_avg if silhouette_avg is not None else 0,
                            "Clustering/Num_Clusters": len(np.unique(patient_clusters)),
                        })
                        # Use summary for clustering method instead of log
                        wandb.run.summary["Clustering_Method"] = "HDBSCAN" if args.use_hdbscan else f"KMeans (k={args.num_clusters})"
                        
                        # Log cluster distribution
                        cluster_counts = np.bincount(patient_clusters[patient_clusters >= 0])
                        cluster_dist_data = [[i, count] for i, count in enumerate(cluster_counts)]
                        if -1 in np.unique(patient_clusters):  # Add noise points if any
                            noise_count = np.sum(patient_clusters == -1)
                            # Use -1 as the label for noise instead of the string "noise" to maintain consistent types
                            cluster_dist_data.append([-1, noise_count])
                            
                        wandb.log({"Clustering/Distribution": wandb.Table(
                            columns=["cluster", "count"],
                            data=cluster_dist_data
                        )})
                      
                    # Save locally as well
                    cluster_plot_path = os.path.join(args.output_dir, f'clustering_visualization_{args.cancer_type}.png')
                    plt.savefig(cluster_plot_path)
                    plt.close()
                    print(f"Clustering visualization saved to {cluster_plot_path}")
                    
            except Exception as e:
                print(f"Error during clustering: {e}")
                patient_clusters = None
                silhouette_avg = None
    else:
        print("Skipping clustering: no patient embeddings available")

    # --- Survival Analysis (only if clustering succeeded) --- #
    if patient_clusters is not None and patient_ids is not None:
         run_survival_analysis(patient_ids, patient_clusters, clinical_df, args.output_dir, args.cancer_type)
    else:
         print("Skipping survival analysis because clustering assignments are not available.")

    # --- Saving Results --- #
    if args.output_dir:
        # Ensure directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        results = {
            'patient_ids': patient_ids,
            'cluster_assignments': patient_clusters,
            'final_patient_embeddings_gcn': final_patient_embeddings_gcn,
            'tsne_embeddings': tsne_embeddings,
            'gene_list': gene_list,
            'silhouette_score': silhouette_avg,
            'args': vars(args)
        }
        results_save_path = os.path.join(args.output_dir, f'{args.model_type}_evaluation_results_{args.cancer_type}.joblib')
        try:
             joblib.dump(results, results_save_path)
             print(f"Evaluation results saved to {results_save_path}")
             
             # Upload results as an artifact to wandb
             if wandb.run and wandb.run.mode != "disabled":
                 results_artifact = wandb.Artifact(
                     name=f"{args.model_type}-eval-results-{args.cancer_type}",
                     type="evaluation_results",
                     description=f"Evaluation results for {args.model_type} on {args.cancer_type}"
                 )
                 results_artifact.add_file(results_save_path)
                 wandb.log_artifact(results_artifact)
        except Exception as e:
             print(f"Error saving results to joblib: {e}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    # Log final metrics and finish wandb run
    if wandb.run and wandb.run.mode != "disabled":
        wandb.log({
            "Performance/Execution_Time_Seconds": execution_time
        })
        # Use summary for completion status instead of log
        wandb.run.summary["Successful_Completion"] = True
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation and Visualization for GCN Patient Stratification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Input Data --- #
    parser.add_argument('--model_type', type=str, required=True, choices=['autoencoder', 'gcn', 'integrated'],
                        help='Type of model/embeddings to evaluate (determines default paths)')
    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Path to pre-computed embeddings .joblib file. If not provided, will use default path based on model_type.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (.pth file). Will run inference if provided.')
    parser.add_argument('--original_data_path', type=str, default=None,
                        help='Path to original prepared_data .joblib (for graph structure, omics links). Required if model_path is provided.')
    parser.add_argument('--clinical_data_path', type=str, default=None,
                        help='Optional path to clinical data file (CSV/TSV) for survival analysis. Requires columns like "patient_id", "duration", "event".')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type key in the prepared_data file.')

    # --- Graph Construction (if using model_path) --- #
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

    # --- Model Architecture (if using model_path) --- #
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

    # --- Autoencoder Architecture --- #
    ae_model_group = parser.add_argument_group('Autoencoder Model Architecture')
    ae_model_group.add_argument('--gene_embedding_dim', type=int, default=64,
                      help='Dimension of gene embeddings in the autoencoder.')
    ae_model_group.add_argument('--patient_embedding_dim', type=int, default=128,
                      help='Dimension of patient embeddings in the autoencoder.')
    ae_model_group.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for processing data with autoencoder.')

    # --- Clustering --- #
    cluster_group = parser.add_argument_group('Clustering')
    cluster_group.add_argument('--use_existing_clusters', action='store_true',
                      help='Use cluster assignments from embedding file if available (skip clustering).')
    cluster_group.add_argument('--num_clusters', type=int, default=3,
                        help='Number of patient clusters for KMeans (if HDBSCAN is not used or fails).')
    # TSNE parameters
    cluster_group.add_argument('--tsne_components', type=int, default=2, 
                        help='Number of components for TSNE dimensionality reduction.')
    cluster_group.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help='Perplexity parameter for TSNE.')
    
    # HDBSCAN parameters
    cluster_group.add_argument('--use_hdbscan', action='store_true',
                        help='Use HDBSCAN for clustering instead of KMeans.')
    cluster_group.add_argument('--hdbscan_min_cluster_size', type=int, default=-1,
                        help='Minimum cluster size for HDBSCAN. If <= 0, automatically set to 5% of data size.')
    cluster_group.add_argument('--hdbscan_min_samples', type=int, default=5,
                        help='Min samples parameter for HDBSCAN (controls cluster density).')
    cluster_group.add_argument('--hdbscan_epsilon', type=float, default=0.0,
                        help='Cluster selection epsilon for HDBSCAN. 0 means automatic determination.')

    # --- Output & Misc --- #
    output_group = parser.add_argument_group('Output & Miscellaneous')
    output_group.add_argument('--output_dir', type=str, default='./gcn_evaluation_results',
                        help='Directory to save evaluation results.')
    output_group.add_argument('--force_proceed_on_validation_error', action='store_true',
                       help='Attempt to proceed even if HeteroData validation fails.')

    args = parser.parse_args()
    
    # --- Validate Arguments --- #
    if args.embedding_path is None and args.model_path is None:
        parser.error("Either --embedding_path or --model_path must be provided")
    
    if args.model_path is not None and args.original_data_path is None:
        parser.error("--original_data_path is required when using --model_path")
    
    # --- Create output directory --- #
    if args.output_dir:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add key info to directory name
        method = "hdbscan" if args.use_hdbscan else f"kmeans{args.num_clusters}"
        source = "model" if args.model_path else "embed"
        run_name = f"{args.model_type}_{args.cancer_type}_{source}_{method}_{time_stamp}"
        args.output_dir = os.path.join(args.output_dir, run_name)
        try:
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Created output directory: {args.output_dir}")
        except OSError as e:
             print(f"Error creating output directory {args.output_dir}: {e}")
             # Fallback to a default name if creation fails
             args.output_dir = f"./evaluation_results/{args.model_type}_run_{time_stamp}" 
             os.makedirs(args.output_dir, exist_ok=True)
             print(f"Using fallback output directory: {args.output_dir}")
    
    run_evaluation(args)

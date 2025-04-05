import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch_geometric.utils import dense_to_sparse

def load_prepared_data(filepath):
    """Loads the prepared data dictionary from a joblib file."""
    try:
        data = joblib.load(filepath)
        print(f"Successfully loaded data from {filepath}")
        # Example: Print keys for the first cancer type found
        if data:
            first_cancer = next(iter(data))
            print(f"Keys for '{first_cancer}': {list(data[first_cancer].keys())}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

# --- Replace OmicsDataset with JointOmicsDataset ---
class JointOmicsDataset(Dataset):
    """PyTorch Dataset for structured omics data (genes x modalities)."""
    def __init__(self, omics_data_dict, gene_list, modalities=['rnaseq', 'methylation', 'scnv', 'miRNA']):
        """
        Args:
            omics_data_dict (dict): Dictionary where keys are omics types and values are pandas
                                    DataFrames (patients x features).
            gene_list (list): The master list of genes defining the feature order.
            modalities (list): List of omics modalities to include, in desired order.
        """
        self.gene_list = list(gene_list)
        self.modalities = modalities
        self.num_genes = len(self.gene_list)
        self.num_modalities = len(self.modalities)

        processed_omics = {} # Store processed dataframes
        patient_ids = None

        print(f"Processing omics data for modalities: {self.modalities}")

        for omics_type in self.modalities:
            if omics_type in omics_data_dict:
                print(f"  Processing {omics_type}...")
                df = omics_data_dict[omics_type].copy()

                # Ensure 'patient_id' is the index
                if 'patient_id' in df.columns:
                    # Check for duplicate patient IDs before setting index
                    if df['patient_id'].duplicated().any():
                         print(f"    Warning: Duplicate patient IDs found in {omics_type}. Keeping first occurrence.")
                         df = df.drop_duplicates(subset=['patient_id'], keep='first')
                    df = df.set_index('patient_id')
                elif df.index.name == 'patient_id':
                     pass # Already indexed correctly
                else:
                    print(f"    Warning: Could not identify patient_id index/column in {omics_type}. Assuming index is patient ID.")

                # Align columns (genes) with the master gene_list
                missing_cols = list(set(self.gene_list) - set(df.columns))
                if missing_cols:
                     print(f"    Adding {len(missing_cols)} missing gene columns filled with 0.0")
                     for col in missing_cols:
                         df[col] = 0.0
                # Ensure all gene_list columns exist before reindexing
                cols_to_keep = [g for g in self.gene_list if g in df.columns]
                df = df[cols_to_keep]
                # Add any remaining genes from gene_list that might have been missed (shouldn't happen if previous step worked)
                final_missing_cols = list(set(self.gene_list) - set(df.columns))
                if final_missing_cols:
                    print(f"    Adding {len(final_missing_cols)} final missing gene columns filled with 0.0")
                    for col in final_missing_cols:
                        df[col] = 0.0
                # Reorder and select columns to match gene_list exactly
                df = df[self.gene_list]

                # Convert to float32 for efficiency
                df = df.astype(np.float32)

                # Align patients
                if patient_ids is None:
                    patient_ids = df.index
                else:
                    # Keep only common patients across modalities
                    common_patients = patient_ids.intersection(df.index)
                    if len(common_patients) < len(patient_ids):
                        print(f"    Aligning patients: {len(patient_ids)} -> {len(common_patients)}")
                        patient_ids = common_patients
                        # Filter previously processed dataframes
                        for p_type in processed_omics:
                            processed_omics[p_type] = processed_omics[p_type].loc[patient_ids]
                    # Filter current dataframe
                    df = df.loc[patient_ids]

                processed_omics[omics_type] = df
                print(f"    Processed {omics_type} shape: {df.shape}")

            else:
                print(f"  Warning: Modality '{omics_type}' not found in omics_data_dict. Skipping.")

        if not processed_omics:
            raise ValueError("No valid omics data found for the specified modalities.")
        if patient_ids is None or patient_ids.empty:
             raise ValueError("No common patients found across the specified modalities.")

        self.processed_omics = processed_omics # Dict of DFs (patients x genes)
        self.patient_ids = patient_ids.tolist()
        self.modalities_in_data = list(processed_omics.keys()) # Actual modalities used
        self.num_modalities = len(self.modalities_in_data)

        print(f"\nDataset initialized with {len(self.patient_ids)} patients, {self.num_genes} genes, {self.num_modalities} modalities.")
        print(f"Final modalities included: {self.modalities_in_data}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        # Retrieve the row for this patient from each modality's DataFrame
        # This results in a list of Pandas Series (one per modality), each of length num_genes
        patient_data_series = [self.processed_omics[omics_type].loc[patient_id] for omics_type in self.modalities_in_data]

        # Convert each Series to a NumPy array and stack them
        # Stack along axis 0 -> shape (num_modalities, num_genes)
        stacked_data_np = np.stack([s.values for s in patient_data_series], axis=0)

        # Convert to tensor -> shape (num_modalities, num_genes)
        tensor_data = torch.tensor(stacked_data_np, dtype=torch.float32)

        # Transpose to get shape (num_genes, num_modalities)
        return tensor_data.t()

def prepare_graph_data(adj_matrix, 
                       gene_list=None, 
                       omics_data_dict=None, 
                       node_init_modality='identity'):
    """Prepares graph data for the InteractionGraphAutoencoder.

    Args:
        adj_matrix (sparse or dense matrix): The adjacency matrix.
        gene_list (list, optional): Ordered list of gene names corresponding to nodes.
                                  Required if node_init_modality is not 'identity'.
        omics_data_dict (dict, optional): Dictionary of omics DataFrames (patient x gene).
                                      Required if node_init_modality is not 'identity'.
        node_init_modality (str): Specifies how to initialize node features.
                                 Options: 'identity', 'rnaseq', 'methylation', etc.
                                 Defaults to 'identity'.

    Returns:
        tuple: Contains:
            - node_features (Tensor): Initial node features (num_nodes, feature_dim).
            - edge_index (LongTensor): Graph connectivity (2, num_edges).
            - edge_weight (Tensor): Edge weights (num_edges,).
            - adj_tensor (Tensor): Original adjacency matrix as a dense tensor.
    """
    num_nodes = adj_matrix.shape[0]

    # Convert adjacency matrix to edge_index format for PyG
    if not isinstance(adj_matrix, torch.Tensor):
        # If sparse, convert to dense first for dense_to_sparse
        if hasattr(adj_matrix, "toarray"):
            adj_matrix_dense = adj_matrix.toarray()
        else:
            adj_matrix_dense = np.asarray(adj_matrix) # Handle regular numpy array
        adj_tensor = torch.tensor(adj_matrix_dense, dtype=torch.float32)
    else:
        adj_tensor = adj_matrix.float() # Assume it's already dense if tensor
        if adj_tensor.is_sparse:
            adj_tensor = adj_tensor.to_dense()

    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    print(f"Converted adj matrix ({adj_tensor.shape}) to edge_index ({edge_index.shape}) and edge_weight ({edge_weight.shape})")

    # Prepare initial node features
    if node_init_modality == 'identity':
        node_features = torch.eye(num_nodes, dtype=torch.float32)
        print(f"Using identity matrix for node features. Shape: {node_features.shape}")
    elif omics_data_dict is not None and gene_list is not None:
        if node_init_modality in omics_data_dict:
            print(f"Using average '{node_init_modality}' for node features...")
            omics_df = omics_data_dict[node_init_modality]
            
            # Ensure the dataframe is aligned with gene_list (should be done in dataset creation, but verify)
            if not all(g in omics_df.columns for g in gene_list):
                 raise ValueError(f"Omics data '{node_init_modality}' is missing genes from gene_list.")
            if list(omics_df.columns) != gene_list:
                 print(f"  Reordering omics columns to match gene_list for node features.")
                 omics_df = omics_df[gene_list]
            
            # Calculate mean expression per gene across patients
            # Ensure the index is patient_id if not already set (handle potential issues)
            if 'patient_id' in omics_df.columns:
                omics_df = omics_df.set_index('patient_id')
            elif omics_df.index.name != 'patient_id':
                 print(f"  Warning: Assuming index of '{node_init_modality}' is patient ID for averaging.")

            mean_features = omics_df.mean(axis=0).fillna(0).values # Calculate mean, fill NaNs with 0
            
            if len(mean_features) != num_nodes:
                raise ValueError(f"Number of features ({len(mean_features)}) does not match number of nodes ({num_nodes}).")
                
            # Reshape to (num_nodes, 1)
            node_features = torch.tensor(mean_features, dtype=torch.float32).unsqueeze(1)
            print(f"Using average {node_init_modality} for node features. Shape: {node_features.shape}")
        else:
            raise ValueError(f"Modality '{node_init_modality}' not found in omics_data_dict.")
    else:
         # This case occurs if modality is not 'identity' but omics_data or gene_list is missing
         raise ValueError("omics_data_dict and gene_list must be provided for non-identity node features.")

    # Return edge_weight along with other components
    return node_features, edge_index, edge_weight, adj_tensor

if __name__ == '__main__':
    # Example usage: Load data for both cancer types
    data_path = '../../data/prepared_data_both.joblib' # Assuming relative path from modelling/autoencoder/
    prepared_data = load_prepared_data(data_path)

    if prepared_data and 'colorec' in prepared_data:
        print("\n--- Processing Colorectal Cancer Data for Joint Dataset---")
        colorec_data = prepared_data['colorec']
        omics_data_colorec = colorec_data['omics_data'] # Get omics data
        adj_matrix_colorec = colorec_data['adj_matrix']
        gene_list_colorec = colorec_data['gene_list']

        # Example 1: Identity features
        print("\nTesting identity features:")
        graph_node_features_id, graph_edge_index_id, graph_edge_weight_id, graph_adj_tensor_id = prepare_graph_data(
            adj_matrix_colorec, 
            node_init_modality='identity' # Explicitly request identity
        )
        print(f"Identity node features shape: {graph_node_features_id.shape}")
        
        # Example 2: RNA-seq features
        print("\nTesting rnaseq features:")
        if 'rnaseq' in omics_data_colorec:
            graph_node_features_rna, graph_edge_index_rna, graph_edge_weight_rna, graph_adj_tensor_rna = prepare_graph_data(
                adj_matrix_colorec, 
                gene_list=gene_list_colorec, # Pass gene list
                omics_data_dict=omics_data_colorec, # Pass omics data
                node_init_modality='rnaseq' # Request rnaseq
            )
            print(f"RNA-seq node features shape: {graph_node_features_rna.shape}")
            # print(f"Sample RNA-seq features:\n{graph_node_features_rna[:5]}") # Optional: print sample
        else:
             print("Skipping RNA-seq feature test: 'rnaseq' data not found.")

        # Prepare Joint Omics Data (no change here)
        modalities_to_use = ['rnaseq', 'methylation', 'scnv', 'miRNA']

        try:
            joint_omics_dataset = JointOmicsDataset(omics_data_colorec, gene_list_colorec, modalities=modalities_to_use)
            print(f"\nCreated JointOmicsDataset with {len(joint_omics_dataset)} patients.")

            # Example: Get the first patient's data
            if len(joint_omics_dataset) > 0:
                first_patient_structured_features = joint_omics_dataset[0]
                print(f"\nShape of structured features for first patient (genes x modalities): {first_patient_structured_features.shape}")
                print(f"Expected shape: ({joint_omics_dataset.num_genes}, {joint_omics_dataset.num_modalities})")
                # Check a few values
                print("Sample values from first patient tensor [gene 0:5, modality 0]:")
                print(first_patient_structured_features[:5, 0])
                if joint_omics_dataset.num_modalities > 1:
                    print("Sample values from first patient tensor [gene 0:5, modality 1]:")
                    print(first_patient_structured_features[:5, 1])

            # Example of creating DataLoader
            # joint_dataloader = DataLoader(joint_omics_dataset, batch_size=16, shuffle=True)
            # print(f"\nCreated DataLoader with {len(joint_dataloader)} batches.")
            # for batch in joint_dataloader:
            #    print(f"Batch shape: {batch.shape}") # Expected: [batch_size, num_genes, num_modalities]
            #    break

        except ValueError as e:
            print(f"Error creating JointOmicsDataset: {e}")
        except KeyError as e:
             print(f"Error creating JointOmicsDataset: Missing key {e}")

    if prepared_data and 'panc' in prepared_data:
        print("\n--- Processing Pancreatic Cancer Data ---")
        # Repeat similar processing for pancreatic cancer data
        panc_data = prepared_data['panc']
        # ... (add similar checks and dataset creation for panc)
        pass 
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

def prepare_graph_data(adj_matrix, use_identity_features=True):
    """Prepares graph data for the InteractionGraphAutoencoder."""
    num_nodes = adj_matrix.shape[0]

    # Convert adjacency matrix to edge_index format for PyG
    # Ensure input is a torch tensor
    if not isinstance(adj_matrix, torch.Tensor):
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    else:
        adj_tensor = adj_matrix.float()

    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    # We might not need edge_weight if the GCN model doesn't use it explicitly,
    # but dense_to_sparse returns it. We primarily need edge_index.
    print(f"Converted adj matrix ({adj_tensor.shape}) to edge_index ({edge_index.shape})")

    # Prepare initial node features
    if use_identity_features:
        node_features = torch.eye(num_nodes, dtype=torch.float32)
        print(f"Using identity matrix for node features. Shape: {node_features.shape}")
    else:
        # If you have other node features, prepare them here
        # node_features = ... # Shape: (num_nodes, feature_dim)
        raise NotImplementedError("Custom node features not implemented yet.")
        # print(f"Using custom node features. Shape: {node_features.shape}")

    return node_features, edge_index, adj_tensor

if __name__ == '__main__':
    # Example usage: Load data for both cancer types
    data_path = '../../data/prepared_data_both.joblib' # Assuming relative path from modelling/autoencoder/
    prepared_data = load_prepared_data(data_path)

    if prepared_data and 'colorec' in prepared_data:
        print("\n--- Processing Colorectal Cancer Data for Joint Dataset---")
        colorec_data = prepared_data['colorec']

        # 1. Prepare Graph Data (still needed for graph AE part)
        adj_matrix_colorec = colorec_data['adj_matrix']
        gene_list_colorec = colorec_data['gene_list']
        graph_node_features, graph_edge_index, graph_adj_tensor = prepare_graph_data(adj_matrix_colorec)

        # 2. Prepare Joint Omics Data
        omics_data_colorec = colorec_data['omics_data']
        modalities_to_use = ['rnaseq', 'methylation', 'scnv', 'miRNA'] # Define desired order

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
import os
import pandas as pd
import numpy as np
from scipy import sparse
import json
import pickle

class InteractionProcessor:
    """
    A class for processing gene interaction data from BioGRID into a format suitable
    for Graph Convolutional Networks (GCNs).
    """

    def __init__(self, interaction_dir, gene_list=None, identifier_type='Official Symbol'):
        """
        Initialize the InteractionProcessor.

        Args:
            interaction_dir (str): Directory containing interaction CSV files
            gene_list (list, optional): List of "core" genes from omics data
            identifier_type (str): Type of gene identifier to use. Options include 
                                   'Official Symbol', 'Systematic Name', or 'Synonym'
        """
        self.interaction_dir = interaction_dir
        self.core_gene_list = gene_list if gene_list else []
        self.identifier_type = identifier_type
        self.gene_mapping = {}
        self.node_list = []
        self.core_gene_indices = set()
        self.adjacency_matrix = None
        self.processed_dir = os.path.join(interaction_dir, "processed")
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_interactions(self, interaction_file):
        """
        Load interaction data from a CSV file.
        
        Args:
            interaction_file (str): Path to the interaction CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing the interaction data
        """
        print(f"Loading interaction data from {interaction_file}...")
        try:
            df = pd.read_csv(interaction_file)
            print(f"Loaded {len(df)} interactions.")
            return df
        except Exception as e:
            print(f"Error loading interaction file: {e}")
            return None

    def identify_unique_genes(self, interactions_df):
        """
        Extract all unique genes from interaction data.
        
        Args:
            interactions_df (pd.DataFrame): DataFrame containing interaction data
            
        Returns:
            set: Set of unique gene identifiers
        """
        print(f"Identifying unique genes using {self.identifier_type}...")
        
        # Set column names based on identifier type
        if self.identifier_type == 'Official Symbol':
            col_a = 'Official Symbol Interactor A'
            col_b = 'Official Symbol Interactor B'
        elif self.identifier_type == 'Systematic Name':
            col_a = 'Systematic Name Interactor A'
            col_b = 'Systematic Name Interactor B'
        elif self.identifier_type == 'Entrez Gene Interactor':
            col_a = 'Entrez Gene Interactor A'
            col_b = 'Entrez Gene Interactor B'
        else:
            raise ValueError(f"Unsupported identifier type: {self.identifier_type}")
        
        # Check if columns exist
        if col_a not in interactions_df.columns or col_b not in interactions_df.columns:
            available_cols = interactions_df.columns.tolist()
            raise ValueError(f"Identifier columns not found. Available columns: {available_cols}")
        
        # Extract unique genes from both interactor columns
        genes_a = set(interactions_df[col_a].dropna().unique())
        genes_b = set(interactions_df[col_b].dropna().unique())
        
        # Combine and remove any placeholders or empty values
        all_genes = genes_a.union(genes_b)
        all_genes = {gene for gene in all_genes if gene and gene != '-'}
        
        print(f"Identified {len(all_genes)} unique genes.")
        return all_genes

    def create_gene_mapping(self, unique_genes):
        """
        Create a mapping from gene identifiers to integer indices.
        
        Args:
            unique_genes (set): Set of unique gene identifiers
            
        Returns:
            dict: Mapping from gene identifiers to indices
        """
        print("Creating gene-to-index mapping...")
        self.gene_mapping = {gene: idx for idx, gene in enumerate(sorted(unique_genes))}
        self.node_list = sorted(unique_genes)
        
        print(f"Created mapping for {len(self.gene_mapping)} genes.")
        return self.gene_mapping

    def identify_core_gene_indices(self):
        """
        Identify the indices of core genes in the gene mapping.
        
        Returns:
            set: Set of indices corresponding to core genes
        """
        print("Identifying core gene indices...")
        self.core_gene_indices = set()
        
        # Count how many core genes are found in the mapping
        found_count = 0
        
        for gene in self.core_gene_list:
            if gene in self.gene_mapping:
                self.core_gene_indices.add(self.gene_mapping[gene])
                found_count += 1
        
        coverage_percent = (found_count / len(self.core_gene_list) * 100) if self.core_gene_list else 0
        print(f"Found {found_count} of {len(self.core_gene_list)} core genes ({coverage_percent:.2f}%).")
        
        return self.core_gene_indices

    def build_adjacency_matrix(self, interactions_df, add_self_loops=True, normalize=True):
        """
        Build an adjacency matrix from the interaction data.
        
        Args:
            interactions_df (pd.DataFrame): DataFrame containing interaction data
            add_self_loops (bool): Whether to add self-loops to the adjacency matrix
            normalize (bool): Whether to normalize the adjacency matrix
            
        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix
        """
        print("Building adjacency matrix...")
        
        # Set column names based on identifier type
        if self.identifier_type == 'Official Symbol':
            col_a = 'Official Symbol Interactor A'
            col_b = 'Official Symbol Interactor B'
        elif self.identifier_type == 'Systematic Name':
            col_a = 'Systematic Name Interactor A'
            col_b = 'Systematic Name Interactor B'
        elif self.identifier_type == 'Entrez Gene Interactor':
            col_a = 'Entrez Gene Interactor A'
            col_b = 'Entrez Gene Interactor B'
        else:
            raise ValueError(f"Unsupported identifier type: {self.identifier_type}")
        
        # Initialize lists for row and column indices
        rows = []
        cols = []
        
        # Iterate through interactions and add edges
        edge_count = 0
        
        for _, row in interactions_df.iterrows():
            gene_a = row[col_a]
            gene_b = row[col_b]
            
            # Skip if either gene is missing or invalid
            if pd.isna(gene_a) or pd.isna(gene_b) or gene_a == '-' or gene_b == '-':
                continue
            
            # Skip if either gene is not in the mapping (should not happen, but check anyway)
            if gene_a not in self.gene_mapping or gene_b not in self.gene_mapping:
                continue
            
            # Get indices
            idx_a = self.gene_mapping[gene_a]
            idx_b = self.gene_mapping[gene_b]
            
            # Add edges in both directions (undirected graph)
            rows.extend([idx_a, idx_b])
            cols.extend([idx_b, idx_a])
            
            edge_count += 1
        
        # Create data values (all 1s for unweighted graph)
        data = np.ones(len(rows), dtype=np.float32)
        
        # Create sparse adjacency matrix
        n_nodes = len(self.gene_mapping)
        adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # Convert to CSR format for efficient operations
        adj = adj.tocsr()
        
        # Remove duplicate edges
        adj.eliminate_zeros()
        
        print(f"Created adjacency matrix with {edge_count} edges between {n_nodes} nodes.")
        
        # Add self-loops if requested
        if add_self_loops:
            print("Adding self-loops to adjacency matrix...")
            # Add identity matrix to add self-loops
            adj = adj + sparse.eye(n_nodes, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            print("Normalizing adjacency matrix...")
            # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
            # Get degree matrix D (as array)
            degrees = np.array(adj.sum(axis=1)).flatten()
            
            # Calculate D^(-1/2)
            with np.errstate(divide='ignore'):
                d_inv_sqrt = np.power(degrees, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0
            
            # Create diagonal matrix D^(-1/2)
            d_inv_sqrt_mat = sparse.diags(d_inv_sqrt)
            
            # Calculate D^(-1/2) * A * D^(-1/2)
            adj = d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat
        
        self.adjacency_matrix = adj
        return adj

    def save_outputs(self, prefix):
        """
        Save the preprocessed outputs to files.
        
        Args:
            prefix (str): Prefix for output filenames
            
        Returns:
            dict: Paths to saved files
        """
        print(f"Saving preprocessed outputs with prefix '{prefix}'...")
        
        # Create paths
        adj_path = os.path.join(self.processed_dir, f"{prefix}_adjacency.npz")
        mapping_path = os.path.join(self.processed_dir, f"{prefix}_gene_mapping.json")
        node_list_path = os.path.join(self.processed_dir, f"{prefix}_node_list.json")
        core_indices_path = os.path.join(self.processed_dir, f"{prefix}_core_indices.pkl")
        metadata_path = os.path.join(self.processed_dir, f"{prefix}_metadata.json")
        
        # Save adjacency matrix
        sparse.save_npz(adj_path, self.adjacency_matrix)
        
        # Save gene mapping
        with open(mapping_path, 'w') as f:
            json.dump(self.gene_mapping, f)
        
        # Save node list
        with open(node_list_path, 'w') as f:
            json.dump(self.node_list, f)
        
        # Save core gene indices
        with open(core_indices_path, 'wb') as f:
            pickle.dump(list(self.core_gene_indices), f)
        
        # Save metadata
        metadata = {
            'num_nodes': len(self.node_list),
            'num_edges': int(self.adjacency_matrix.sum()) // 2,  # Divide by 2 for undirected graph
            'num_core_genes': len(self.core_gene_indices),
            'identifier_type': self.identifier_type,
            'has_self_loops': True,  # Assuming save is called after build_adjacency_matrix with default params
            'is_normalized': True,    # Assuming save is called after build_adjacency_matrix with default params
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print("Successfully saved all outputs.")
        
        return {
            'adjacency_matrix': adj_path,
            'gene_mapping': mapping_path,
            'node_list': node_list_path,
            'core_gene_indices': core_indices_path,
            'metadata': metadata_path
        }

    def process_interactions(self, interaction_file, prefix, add_self_loops=True, normalize=True):
        """
        Process interactions from start to finish.
        
        Args:
            interaction_file (str): Path to the interaction CSV file
            prefix (str): Prefix for output filenames
            add_self_loops (bool): Whether to add self-loops to the adjacency matrix
            normalize (bool): Whether to normalize the adjacency matrix
            
        Returns:
            dict: Paths to saved files or None if processing failed
        """
        try:
            # Load interactions
            interactions_df = self.load_interactions(interaction_file)
            if interactions_df is None:
                return None
            
            # Process
            unique_genes = self.identify_unique_genes(interactions_df)
            self.create_gene_mapping(unique_genes)
            self.identify_core_gene_indices()
            self.build_adjacency_matrix(interactions_df, add_self_loops, normalize)
            
            # Save results
            output_paths = self.save_outputs(prefix)
            
            print(f"Successfully processed interactions with prefix '{prefix}'.")
            return output_paths
            
        except Exception as e:
            print(f"Error processing interactions: {e}")
            return None

    def load_processed_data(self, prefix):
        """
        Load previously processed data.
        
        Args:
            prefix (str): Prefix of the files to load
            
        Returns:
            dict: Loaded data or None if loading failed
        """
        try:
            print(f"Loading processed data with prefix '{prefix}'...")
            
            # Create paths
            adj_path = os.path.join(self.processed_dir, f"{prefix}_adjacency.npz")
            mapping_path = os.path.join(self.processed_dir, f"{prefix}_gene_mapping.json")
            node_list_path = os.path.join(self.processed_dir, f"{prefix}_node_list.json")
            core_indices_path = os.path.join(self.processed_dir, f"{prefix}_core_indices.pkl")
            
            # Load adjacency matrix
            self.adjacency_matrix = sparse.load_npz(adj_path)
            
            # Load gene mapping
            with open(mapping_path, 'r') as f:
                self.gene_mapping = json.load(f)
            
            # Load node list
            with open(node_list_path, 'r') as f:
                self.node_list = json.load(f)
            
            # Load core gene indices
            with open(core_indices_path, 'rb') as f:
                self.core_gene_indices = set(pickle.load(f))
            
            print("Successfully loaded all processed data.")
            
            return {
                'adjacency_matrix': self.adjacency_matrix,
                'gene_mapping': self.gene_mapping,
                'node_list': self.node_list,
                'core_gene_indices': self.core_gene_indices
            }
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None 
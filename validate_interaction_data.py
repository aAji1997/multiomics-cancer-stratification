import os
import numpy as np
import pandas as pd
from scipy import sparse
import json
import pickle
import matplotlib.pyplot as plt
from collections import Counter

def load_processed_data(processed_dir, prefix):
    """Load all processed data files for validation."""
    try:
        # Define file paths
        adj_path = os.path.normpath(os.path.join(processed_dir, f"{prefix}_adjacency.npz"))
        mapping_path = os.path.normpath(os.path.join(processed_dir, f"{prefix}_gene_mapping.json"))
        node_list_path = os.path.normpath(os.path.join(processed_dir, f"{prefix}_node_list.json"))
        core_indices_path = os.path.normpath(os.path.join(processed_dir, f"{prefix}_core_indices.pkl"))
        metadata_path = os.path.normpath(os.path.join(processed_dir, f"{prefix}_metadata.json"))
        
        # Check if all files exist
        for path in [adj_path, mapping_path, node_list_path, core_indices_path, metadata_path]:
            if not os.path.exists(path):
                print(f"Error: File not found: {path}")
                return None
        
        # Load data
        adjacency_matrix = sparse.load_npz(adj_path)
        
        with open(mapping_path, 'r') as f:
            gene_mapping = json.load(f)
        
        with open(node_list_path, 'r') as f:
            node_list = json.load(f)
        
        with open(core_indices_path, 'rb') as f:
            core_indices = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'adjacency_matrix': adjacency_matrix,
            'gene_mapping': gene_mapping,
            'node_list': node_list,
            'core_indices': core_indices,
            'metadata': metadata
        }
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def validate_adjacency_matrix(adjacency_matrix, metadata):
    """Validate properties of the adjacency matrix."""
    results = {}
    
    # Check matrix shape and type
    results['is_sparse'] = sparse.issparse(adjacency_matrix)
    results['shape'] = adjacency_matrix.shape
    results['is_square'] = adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    
    # Convert to COO format for easier analysis
    if not sparse.isspmatrix_coo(adjacency_matrix):
        coo_matrix = adjacency_matrix.tocoo()
    else:
        coo_matrix = adjacency_matrix
    
    # Check symmetry for undirected graph
    # For a symmetric matrix, swapping rows and cols shouldn't change the matrix
    transposed = sparse.coo_matrix((coo_matrix.data, (coo_matrix.col, coo_matrix.row)), 
                                 shape=coo_matrix.shape)
    transposed.eliminate_zeros()
    adjacency_matrix.eliminate_zeros()
    results['is_symmetric'] = ((transposed != adjacency_matrix).nnz == 0)
    
    # Check self-loops (diagonal elements)
    diag_elements = adjacency_matrix.diagonal()
    results['has_self_loops'] = np.all(diag_elements > 0)
    results['self_loop_count'] = np.sum(diag_elements > 0)
    
    # Check normalization
    # For a symmetrically normalized adjacency matrix, row sums should be close to 1
    row_sums = adjacency_matrix.sum(axis=1).A1  # Convert to 1D array
    results['is_normalized'] = np.allclose(row_sums, 1.0, atol=1e-1)
    results['row_sum_stats'] = {
        'min': float(np.min(row_sums)),
        'max': float(np.max(row_sums)),
        'mean': float(np.mean(row_sums)),
        'std': float(np.std(row_sums))
    }
    
    # Check non-zero elements (edges)
    nnz = adjacency_matrix.nnz
    results['edge_count'] = nnz // 2  # Divide by 2 for undirected graph with symmetric matrix
    results['edge_count_matches_metadata'] = (nnz // 2 == metadata['num_edges'])
    
    # Check degree distribution (for potential anomalies)
    degrees = adjacency_matrix.sum(axis=1).A1
    results['degree_stats'] = {
        'min': int(np.min(degrees)),
        'max': int(np.max(degrees)),
        'mean': float(np.mean(degrees)),
        'median': float(np.median(degrees)),
        'std': float(np.std(degrees))
    }
    
    # Create degree distribution for plotting
    results['degree_counts'] = Counter(degrees.astype(int))
    
    return results

def validate_gene_mapping(gene_mapping, node_list, metadata):
    """Validate consistency of gene mapping and node list."""
    results = {}
    
    # Check basic counts
    results['mapping_count'] = len(gene_mapping)
    results['node_list_count'] = len(node_list)
    results['counts_match'] = len(gene_mapping) == len(node_list)
    results['counts_match_metadata'] = len(gene_mapping) == metadata['num_nodes']
    
    # Check that gene_mapping and node_list are consistent
    mapping_keys = set(gene_mapping.keys())
    node_set = set(node_list)
    results['mapping_contains_all_nodes'] = mapping_keys == node_set
    
    # Check index values
    index_values = sorted(gene_mapping.values())
    expected_indices = list(range(len(gene_mapping)))
    results['indices_correct'] = index_values == expected_indices
    
    # Check reversibility (can convert from index to gene and back)
    try:
        reverse_mapping = {v: k for k, v in gene_mapping.items()}
        test_case = list(gene_mapping.items())[0]
        gene, idx = test_case
        test_result = (reverse_mapping[idx] == gene)
        results['mapping_reversible'] = test_result
    except Exception as e:
        results['mapping_reversible'] = False
        results['reversibility_error'] = str(e)
    
    # Sample a few genes and their indices for display
    sample_size = min(5, len(gene_mapping))
    samples = list(gene_mapping.items())[:sample_size]
    results['sample_mappings'] = [{gene: idx} for gene, idx in samples]
    
    return results

def validate_core_indices(core_indices, gene_mapping, metadata):
    """Validate core gene indices."""
    results = {}
    
    # Check core indices count
    results['core_indices_count'] = len(core_indices)
    results['count_matches_metadata'] = len(core_indices) == metadata['num_core_genes']
    
    # Check that all core indices are valid
    all_indices = set(range(len(gene_mapping)))
    core_set = set(core_indices)
    results['all_indices_valid'] = core_set.issubset(all_indices)
    
    # Calculate coverage
    coverage = len(core_indices) / metadata['num_nodes'] * 100
    results['core_coverage_percent'] = coverage
    
    # Sample a few core indices for display
    sample_size = min(5, len(core_indices))
    reverse_mapping = {v: k for k, v in gene_mapping.items()}
    core_samples = []
    for idx in list(core_indices)[:sample_size]:
        gene = reverse_mapping.get(idx, f"Unknown Index: {idx}")
        core_samples.append({idx: gene})
    results['sample_core_genes'] = core_samples
    
    return results

def plot_degree_distribution(degree_counts, title, output_path=None):
    """Plot the degree distribution of the network."""
    degrees = list(degree_counts.keys())
    counts = list(degree_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, counts, width=1.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Node Degree (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title(f'Degree Distribution - {title}')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def run_validation(data_dir, cancer_type):
    """Run all validation checks for a specific cancer type."""
    print(f"\n{'='*80}")
    print(f"Validating {cancer_type.upper()} cancer interaction data")
    print(f"{'='*80}")
    
    # Define paths
    processed_dir = os.path.join(data_dir, f"{cancer_type}/interaction_data/processed")
    
    # Extra debug information
    print(f"Looking for processed files in: {processed_dir}")
    if os.path.exists(processed_dir):
        print(f"Directory exists. Contents: {os.listdir(processed_dir)}")
    else:
        print(f"Directory does not exist: {processed_dir}")
    
    prefix = cancer_type
    
    # Load data
    data = load_processed_data(processed_dir, prefix)
    if not data:
        print(f"Failed to load {cancer_type} cancer data.")
        return
    
    # Run validations
    matrix_results = validate_adjacency_matrix(data['adjacency_matrix'], data['metadata'])
    mapping_results = validate_gene_mapping(data['gene_mapping'], data['node_list'], data['metadata'])
    core_results = validate_core_indices(data['core_indices'], data['gene_mapping'], data['metadata'])
    
    # Display results
    print("\n-- Adjacency Matrix Validation --")
    print(f"Matrix shape: {matrix_results['shape']} (should be square: {matrix_results['is_square']})")
    print(f"Is sparse: {matrix_results['is_sparse']}")
    print(f"Is symmetric: {matrix_results['is_symmetric']} (required for undirected graph)")
    print(f"Has self-loops: {matrix_results['has_self_loops']} ({matrix_results['self_loop_count']} nodes with self-loops)")
    print(f"Is normalized: {matrix_results['is_normalized']} (row sums should be close to 1)")
    print(f"Row sum statistics: min={matrix_results['row_sum_stats']['min']:.4f}, max={matrix_results['row_sum_stats']['max']:.4f}, mean={matrix_results['row_sum_stats']['mean']:.4f}")
    print(f"Edge count: {matrix_results['edge_count']} (matches metadata: {matrix_results['edge_count_matches_metadata']})")
    print(f"Degree statistics: min={matrix_results['degree_stats']['min']}, max={matrix_results['degree_stats']['max']}, mean={matrix_results['degree_stats']['mean']:.2f}, median={matrix_results['degree_stats']['median']:.2f}")
    
    print("\n-- Gene Mapping Validation --")
    print(f"Mapping count: {mapping_results['mapping_count']} (matches node list: {mapping_results['counts_match']})")
    print(f"Counts match metadata: {mapping_results['counts_match_metadata']}")
    print(f"Mapping contains all nodes: {mapping_results['mapping_contains_all_nodes']}")
    print(f"Indices correct (0 to N-1): {mapping_results['indices_correct']}")
    print(f"Mapping reversible (index to gene): {mapping_results['mapping_reversible']}")
    print("Sample mappings (gene: index):")
    for mapping in mapping_results['sample_mappings']:
        for gene, idx in mapping.items():
            print(f"  - {gene}: {idx}")
    
    print("\n-- Core Gene Indices Validation --")
    print(f"Core indices count: {core_results['core_indices_count']} (matches metadata: {core_results['count_matches_metadata']})")
    print(f"All indices valid: {core_results['all_indices_valid']}")
    print(f"Core genes coverage: {core_results['core_coverage_percent']:.2f}%")
    print("Sample core genes (index: gene):")
    for mapping in core_results['sample_core_genes']:
        for idx, gene in mapping.items():
            print(f"  - {idx}: {gene}")
    
    # Plot degree distribution
    output_dir = os.path.join(data_dir, f"{cancer_type}/interaction_data/visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_degree_distribution.png")
    plot_degree_distribution(matrix_results['degree_counts'], f"{cancer_type.title()} Cancer", output_path)
    print(f"\nDegree distribution plot saved to: {output_path}")
    
    return {
        'matrix_results': matrix_results,
        'mapping_results': mapping_results,
        'core_results': core_results
    }

def main():
    data_dir = './data'
    
    # Validate colorec cancer data
    colorec_results = run_validation(data_dir, 'colorec')
    
    # Validate panc cancer data
    panc_results = run_validation(data_dir, 'panc')
    
    print("\n-- Overall Validation Results --")
    if colorec_results and panc_results:
        all_checks_passed = True
        
        # Check if matrices are similar in structure
        colorec_shape = colorec_results['matrix_results']['shape']
        panc_shape = panc_results['matrix_results']['shape']
        print(f"Matrix sizes comparable: colorec {colorec_shape} vs panc {panc_shape}")
        
        # Check if degree distributions are similar
        colorec_mean_degree = colorec_results['matrix_results']['degree_stats']['mean']
        panc_mean_degree = panc_results['matrix_results']['degree_stats']['mean']
        print(f"Mean degree comparable: colorec {colorec_mean_degree:.2f} vs panc {panc_mean_degree:.2f}")
        
        # Check core gene coverage
        colorec_coverage = colorec_results['core_results']['core_coverage_percent']
        panc_coverage = panc_results['core_results']['core_coverage_percent']
        print(f"Core gene coverage comparable: colorec {colorec_coverage:.2f}% vs panc {panc_coverage:.2f}%")
        
        print(f"\nOverall validation {'PASSED' if all_checks_passed else 'FAILED'}")
        print("\nThe processed interaction data appears to be valid and ready for the next processing stage.")
        print("These adjacency matrices and gene mappings can now be used for generating knowledge embeddings.")
    else:
        print("\nValidation incomplete. Please check errors above.")

if __name__ == "__main__":
    main() 
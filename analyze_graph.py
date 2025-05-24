import argparse
import os
import joblib
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Local imports (assuming this script is run from the workspace root or similar)
# Need to adjust path if running from a different directory
try:
    # This assumes analyze_graph.py is in the root or a directory from which these imports work
    from modelling.autoencoder.data_utils import load_prepared_data
    from modelling.autoencoder.train_joint_ae import prune_graph # Reuse pruning logic if needed
except ImportError:
    print("Warning: Could not import local modules. Ensure script is run from a path where 'modelling' is accessible, or adjust sys.path.")
    # Define dummy functions if needed for basic execution without local imports
    def load_prepared_data(path):
        print(f"Attempting to load data directly from: {path}")
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Direct load failed: {e}")
            return None
    # Define a basic prune_graph stub if the import fails
    def prune_graph(adj_matrix, **kwargs):
        print("Warning: Using stub prune_graph function. Actual pruning logic not available.")
        return adj_matrix # Return original matrix

def analyze_graph_properties(adj_matrix, gene_list, cancer_type, output_dir):
    """
    Analyzes and visualizes properties of the gene-gene interaction graph.

    Args:
        adj_matrix: The adjacency matrix (scipy sparse or numpy array).
        gene_list: List of gene names corresponding to matrix indices.
        cancer_type: Name of the cancer type for labeling outputs.
        output_dir: Directory to save plots and statistics.
    """
    print(f"\n--- Analyzing Graph for {cancer_type} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Ensure matrix is in COO format for easier processing with NetworkX
    if not sp.isspmatrix_coo(adj_matrix):
        try:
            adj_matrix_coo = sp.coo_matrix(adj_matrix)
            print("Converted adjacency matrix to COO format.")
        except Exception as e:
            print(f"Error converting matrix to COO: {e}")
            return
    else:
        adj_matrix_coo = adj_matrix

    num_nodes = adj_matrix_coo.shape[0]
    num_edges = adj_matrix_coo.nnz
    
    # Check for self-loops
    self_loops = np.sum(adj_matrix_coo.diagonal())
    print(f"Number of self-loops detected: {int(self_loops)} (Note: Edges on the diagonal)")
    
    # Always assume the graph is symmetric (undirected)
    is_symmetric = True
    print("Assuming the graph is symmetric (undirected) for GCN parsing.")
    
    # For undirected graphs: max edges is n*(n-1)/2 (without self-loops)
    possible_edges = (num_nodes * (num_nodes - 1)) / 2
    # If self-loops exist, add them to possible edges
    if self_loops > 0:
        possible_edges += num_nodes
        
    density = num_edges / possible_edges if possible_edges > 0 else 0

    print(f"Number of Nodes (Genes): {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Graph Density: {density:.6f}")

    # --- Create NetworkX graph ---
    # Create graph from the COO matrix, handling potential weights
    # If weights are all 1 or missing, it's unweighted. Otherwise, weighted.
    has_weights = 'weight' in adj_matrix_coo.dtype.names if hasattr(adj_matrix_coo, 'dtype') and adj_matrix_coo.dtype.names else False
    # Always create an undirected graph
    print("Creating Undirected Graph...")
    G = nx.Graph()

    print("Adding edges to NetworkX graph...")
    # Efficiently add edges from COO format
    # If weights exist and are meaningful (not just 1s), add them
    weighted_edges = []
    unweighted_edges = []
    min_weight, max_weight = float('inf'), float('-inf')
    is_weighted = False

    if adj_matrix_coo.nnz > 0:
         # Check if weights are non-trivial
        unique_weights = np.unique(adj_matrix_coo.data)
        if len(unique_weights) > 1 or (len(unique_weights) == 1 and unique_weights[0] != 1):
            is_weighted = True
            min_weight = np.min(adj_matrix_coo.data)
            max_weight = np.max(adj_matrix_coo.data)
            print(f"Graph appears weighted. Min weight: {min_weight:.4f}, Max weight: {max_weight:.4f}")
            for r, c, w in tqdm(zip(adj_matrix_coo.row, adj_matrix_coo.col, adj_matrix_coo.data), total=adj_matrix_coo.nnz, desc="Processing Edges"):
                 # Avoid adding self-loops to NetworkX graph if they exist, unless needed
                 if r != c:
                     weighted_edges.append((r, c, w))
        else:
            print("Graph appears unweighted (or weights are all 1).")
            for r, c in tqdm(zip(adj_matrix_coo.row, adj_matrix_coo.col), total=adj_matrix_coo.nnz, desc="Processing Edges"):
                if r != c:
                    unweighted_edges.append((r,c))

    # Add edges to the graph
    if is_weighted:
        G.add_weighted_edges_from(weighted_edges)
    else:
        G.add_edges_from(unweighted_edges)
        
    # Add nodes explicitly in case some are isolated
    G.add_nodes_from(range(num_nodes))
        
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Degree Analysis ---
    degrees = [d for n, d in G.degree()]
    degree_type = "Degree"

    if degrees: # Check if degrees list is not empty
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        mean_degree = np.mean(degrees)
        median_degree = np.median(degrees)
        std_degree = np.std(degrees)
        print(f"\n{degree_type} Statistics:")
        print(f"  Min: {min_degree}")
        print(f"  Max: {max_degree}")
        print(f"  Mean: {mean_degree:.4f}")
        print(f"  Median: {median_degree}")
        print(f"  Std Dev: {std_degree:.4f}")
        
        # Plot Degree Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(degrees, bins=50, kde=False)
        plt.title(f'{cancer_type} - {degree_type} Distribution')
        plt.xlabel(degree_type)
        plt.ylabel('Number of Nodes')
        plt.yscale('log') # Often helpful for skewed distributions
        plt.grid(axis='y', alpha=0.5)
        plot_path = os.path.join(output_dir, f'{cancer_type}_degree_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved degree distribution plot to {plot_path}")

    else:
        print("No edges found, cannot compute degree statistics.")

    # --- Edge Weight Analysis (if applicable) ---
    if is_weighted:
        weights = adj_matrix_coo.data # Get all non-zero weights
        plt.figure(figsize=(10, 6))
        sns.histplot(weights, bins=50, kde=True)
        plt.title(f'{cancer_type} - Edge Weight Distribution')
        plt.xlabel('Edge Weight')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.5)
        plot_path = os.path.join(output_dir, f'{cancer_type}_edge_weight_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved edge weight distribution plot to {plot_path}")

    # --- Connected Components ---
    components = list(nx.connected_components(G))
    component_type = "Connected Components"

    num_components = len(components)
    print(f"\nNumber of {component_type}: {num_components}")
    if num_components > 0:
        largest_component = max(components, key=len)
        largest_component_size = len(largest_component)
        print(f"Size of Largest Component: {largest_component_size} nodes ({largest_component_size / num_nodes:.2%})")

        # Plot component size distribution (if many components)
        if num_components > 1:
            component_sizes = [len(c) for c in components]
            plt.figure(figsize=(10, 6))
            sns.histplot(component_sizes, bins=min(50, num_components), kde=False)
            plt.title(f'{cancer_type} - {component_type} Size Distribution')
            plt.xlabel('Component Size (Number of Nodes)')
            plt.ylabel('Frequency')
            plt.yscale('log')
            plt.grid(axis='y', alpha=0.5)
            plot_path = os.path.join(output_dir, f'{cancer_type}_component_size_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved component size distribution plot to {plot_path}")
    else:
        print("Graph has no nodes or edges, cannot analyze components.")

    # --- Clustering Coefficient ---
    # This can be computationally expensive for large graphs
    print("\nCalculating Average Clustering Coefficient (may take time)...")
    try:
        # Use average clustering for the whole graph
        avg_clustering = nx.average_clustering(G) # Use weighted version if applicable? nx.average_clustering(G, weight='weight')
        print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        # Note: Transitivity (nx.transitivity(G)) is a global measure related to clustering.
    except Exception as e:
        print(f"Could not calculate average clustering coefficient: {e}")

    # --- Adjacency Matrix Visualization ---
    print("\nGenerating Adjacency Matrix Sparsity Plot...")
    plt.figure(figsize=(8, 8))
    # Use spy for sparse matrices, imshow for dense (but spy is better for large sparse)
    if sp.issparse(adj_matrix):
         # Ensure it's in a format spy handles well, like COO or CSR
         if not sp.isspmatrix_coo(adj_matrix_coo):
             adj_matrix_plot = adj_matrix_coo.tocoo()
         else:
             adj_matrix_plot = adj_matrix_coo
         plt.spy(adj_matrix_plot, markersize=0.1, aspect='auto') # Use small markers
         plt.title(f'{cancer_type} - Adjacency Matrix Sparsity Pattern')
         plt.xlabel('Gene Index')
         plt.ylabel('Gene Index')
    else: # Handle dense matrix if necessary
         plt.imshow(adj_matrix, cmap='Greys', interpolation='nearest')
         plt.title(f'{cancer_type} - Adjacency Matrix')
         plt.xlabel('Gene Index')
         plt.ylabel('Gene Index')
         
    plot_path = os.path.join(output_dir, f'{cancer_type}_adjacency_matrix_sparsity.png')
    plt.savefig(plot_path, dpi=300) # Increase dpi for potentially large matrices
    plt.close()
    print(f"Saved adjacency matrix sparsity plot to {plot_path}")

    # --- Optional: Subgraph Visualization ---
    # Visualizing the full graph is infeasible. Visualize the largest component or a sample.
    if num_nodes > 0 and num_edges > 0 and num_nodes < 500: # Only attempt if graph is reasonably small
        print("\nGenerating Subgraph Visualization (Largest Component if applicable)...")
        try:
             if num_components > 0:
                 subgraph_nodes = largest_component
                 G_sub = G.subgraph(subgraph_nodes)
                 title = f'{cancer_type} - Largest Component Subgraph ({len(subgraph_nodes)} nodes)'
             else: # If only one component (or graph is connected)
                 G_sub = G
                 title = f'{cancer_type} - Graph Visualization ({num_nodes} nodes)'

             plt.figure(figsize=(12, 12))
             # Use a layout algorithm suitable for large graphs, e.g., spring_layout (can be slow) or kamada_kawai_layout
             pos = nx.spring_layout(G_sub, k=0.1, iterations=20) # Adjust parameters as needed
             nx.draw_networkx_nodes(G_sub, pos, node_size=10, alpha=0.7)
             nx.draw_networkx_edges(G_sub, pos, width=0.2, alpha=0.3)
             plt.title(title)
             plt.axis('off')
             plot_path = os.path.join(output_dir, f'{cancer_type}_subgraph_visualization.png')
             plt.savefig(plot_path, dpi=300)
             plt.close()
             print(f"Saved subgraph visualization plot to {plot_path}")
        except Exception as e:
             print(f"Could not generate subgraph visualization: {e}")
    elif num_nodes >= 500:
         print("\nSkipping subgraph visualization because the graph is too large (>500 nodes).")


    # --- Save Summary Statistics ---
    stats = {
        'cancer_type': cancer_type,
        'num_nodes': int(num_nodes),
        'num_edges': int(num_edges),
        'density': density,
        'is_symmetric': is_symmetric,
        'is_weighted': is_weighted,
        'min_weight': float(min_weight) if is_weighted and min_weight != float('inf') else None,
        'max_weight': float(max_weight) if is_weighted and max_weight != float('-inf') else None,
        'num_self_loops': int(self_loops),
        'degree_stats': {
            'type': degree_type,
            'min': int(min_degree) if degrees else None,
            'max': int(max_degree) if degrees else None,
            'mean': float(mean_degree) if degrees else None,
            'median': float(median_degree) if degrees else None,
            'std_dev': float(std_degree) if degrees else None,
        } if degrees else None,
        'num_components': int(num_components),
        'largest_component_size': int(largest_component_size) if num_components > 0 else None,
        'avg_clustering_coefficient': float(avg_clustering) if 'avg_clustering' in locals() else None,
    }

    stats_path = os.path.join(output_dir, f'{cancer_type}_graph_stats.txt')
    with open(stats_path, 'w') as f:
        import json
        f.write(json.dumps(stats, indent=4))
    print(f"Saved graph statistics summary to {stats_path}")

    print(f"--- Analysis Complete for {cancer_type} ---")


def main():
    parser = argparse.ArgumentParser(description='Analyze Gene-Gene Interaction Graph Properties')

    # Data and Paths
    parser.add_argument('--data_path', type=str, default='data/prepared_data_both.joblib',
                        help='Path to the prepared data joblib file relative to workspace root')
    parser.add_argument('--cancer_type', type=str, default='colorec', choices=['colorec', 'panc'],
                        help='Cancer type to analyze')
    parser.add_argument('--output_dir', type=str, default='./graph_analysis_results',
                        help='Directory to save analysis results (plots, stats)')

    # Optional Pruning Arguments (mirroring train script)
    parser.add_argument('--prune_graph', action='store_true',
                       help='Analyze the pruned version of the graph')
    parser.add_argument('--prune_threshold', type=float, default=0.5,
                       help='Edge weight threshold for pruning')
    parser.add_argument('--keep_top_percent', type=float, default=0.1,
                       help='Keep only this percentage of strongest edges (overrides threshold if set)')
    parser.add_argument('--min_edges_per_node', type=int, default=2,
                       help='Ensure each node has at least this many edges after pruning')

    args = parser.parse_args()

    # --- Data Loading ---
    print(f"Loading data from: {args.data_path}")
    prepared_data = load_prepared_data(args.data_path)
    if not prepared_data or args.cancer_type not in prepared_data:
        print(f"Error: Could not load or find data for {args.cancer_type} in {args.data_path}")
        return

    cancer_data = prepared_data[args.cancer_type]
    adj_matrix_orig = cancer_data['adj_matrix'] # Keep original for comparison if pruning
    gene_list = cancer_data['gene_list']

    # Decide which matrix to analyze
    if args.prune_graph:
        print("Pruning graph before analysis...")
        adj_matrix_to_analyze = prune_graph(
            adj_matrix_orig,
            threshold=args.prune_threshold,
            keep_top_percent=args.keep_top_percent,
            min_edges_per_node=args.min_edges_per_node
        )
        analysis_label = f"{args.cancer_type}_pruned"
         # Also analyze the original graph for comparison? Could add a flag for this.
        # print("\nAnalyzing ORIGINAL graph first for comparison...")
        # analyze_graph_properties(adj_matrix_orig, gene_list, f"{args.cancer_type}_original", os.path.join(args.output_dir, f"{args.cancer_type}_original"))
    else:
        adj_matrix_to_analyze = adj_matrix_orig
        analysis_label = f"{args.cancer_type}_original"

    # Define specific output directory for this run
    run_output_dir = os.path.join(args.output_dir, analysis_label)

    # Perform analysis
    analyze_graph_properties(adj_matrix_to_analyze, gene_list, analysis_label, run_output_dir)

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
from preprocessing.data_extractor import DataExtractor

class OmicsAnalyzer(DataExtractor):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def initial_exploration(self, cancer_type):
        print(f"Initial exploration of {cancer_type} omics data:")
        # get the methylation data
        if cancer_type == "colorec":
            methylation = self.harmonized_colorec["methylation"]
        elif cancer_type == "panc":
            methylation = self.harmonized_panc["methylation"]
        # get the number of genes present(columns)
        num_genes = len(methylation.columns) - 1
        print(f"Number of genes in {cancer_type} methylation data: {num_genes}")
        # get the number of samples present(rows)
        num_samples = len(methylation.index)
        print(f"Number of samples in {cancer_type} methylation data: {num_samples}")
        # get the number of unique patients present
        num_patients = methylation["patient_id"].nunique()
        print(f"Number of unique patients in {cancer_type} methylation data: {num_patients}")

        # get the miRNA
        if cancer_type == "colorec":
            miRNA = self.harmonized_colorec["miRNA"]
        elif cancer_type == "panc":
            miRNA = self.harmonized_panc["miRNA"]
        # get the number of miRNAs present(columns)
        num_miRNAs = len(miRNA.columns) - 1
        print(f"Number of miRNAs in {cancer_type} miRNA data: {num_miRNAs}")
        # get the number of samples present(rows)
        num_samples = len(miRNA.index)
        print(f"Number of samples in {cancer_type} miRNA data: {num_samples}")
        # get the number of unique patients present
        num_patients = miRNA["patient_id"].nunique()
        print(f"Number of unique patients in {cancer_type} miRNA data: {num_patients}")

        # get the rnaseq data
        if cancer_type == "colorec":
            rnaseq = self.harmonized_colorec["rnaseq"]
        elif cancer_type == "panc":
            rnaseq = self.harmonized_panc["rnaseq"]
        # get the number of RNA sequences present(columns)
        num_rnaseq = len(rnaseq.columns) - 1
        print(f"Number of RNA sequences in {cancer_type} rnaseq data: {num_rnaseq}")
        # get the number of samples present(rows)
        num_samples = len(rnaseq.index)
        print(f"Number of samples in {cancer_type} rnaseq data: {num_samples}")
        # get the number of unique patients present
        num_patients = rnaseq["patient_id"].nunique()
        print(f"Number of unique patients in {cancer_type} rnaseq data: {num_patients}")

        # get the scnv data
        if cancer_type == "colorec":
            scnv = self.harmonized_colorec["scnv"]
        elif cancer_type == "panc":
            scnv = self.harmonized_panc["scnv"]
        # get the number of SCNV present(columns)
        num_scnv = len(scnv.columns) - 1
        print(f"Number of SCNV in {cancer_type} scnv data: {num_scnv}")
        # get the number of samples present(rows)
        num_samples = len(scnv.index)
        print(f"Number of samples in {cancer_type} scnv data: {num_samples}")
        # get the number of unique patients present
        num_patients = scnv["patient_id"].nunique()
        print(f"Number of unique patients in {cancer_type} scnv data: {num_patients}")

    def harmonize_omics(self, cancer_type):
        if cancer_type == "colorec":
            clinical = self.colorec_omics["clinical"]
            methylation = self.colorec_omics["methylation"]
            miRNA = self.colorec_omics["miRNA"]
            rnaseq = self.colorec_omics["rnaseq"]
            scnv = self.colorec_omics["scnv"]
        elif cancer_type == "panc":
            clinical = self.panc_omics["clinical"]
            methylation = self.panc_omics["methylation"]
            miRNA = self.panc_omics["miRNA"]
            rnaseq = self.panc_omics["rnaseq"]
            scnv = self.panc_omics["scnv"]
        # get the patient_ids that are present in all omics data
        common_patient_ids = set(clinical["patient_id"]) & set(methylation["patient_id"]) & set(miRNA["patient_id"]) & set(rnaseq["patient_id"]) & set(scnv["patient_id"])
        # filter the dataframes to only include the common patient_ids
        clinical = clinical[clinical["patient_id"].isin(common_patient_ids)]
        methylation = methylation[methylation["patient_id"].isin(common_patient_ids)]
        miRNA = miRNA[miRNA["patient_id"].isin(common_patient_ids)]
        rnaseq = rnaseq[rnaseq["patient_id"].isin(common_patient_ids)]
        scnv = scnv[scnv["patient_id"].isin(common_patient_ids)]
        # print the number of patients present in the harmonized dataframes
        #print(f"Number of patients in {cancer_type} harmonized clinical data: {len(clinical)}")
        #print(f"Number of patients in {cancer_type} harmonized methylation data: {len(methylation)}")
        #print(f"Number of patients in {cancer_type} harmonized miRNA data: {len(miRNA)}")
        #print(f"Number of patients in {cancer_type} harmonized rnaseq data: {len(rnaseq)}")
        #print(f"Number of patients in {cancer_type} harmonized scnv data: {len(scnv)}")
        # return the harmonized dataframes in a dictionary
        return {
            "clinical": clinical,
            "methylation": methylation,
            "miRNA": miRNA,
            "rnaseq": rnaseq,
            "scnv": scnv
        }
        
    def visualize_interaction_network(self, cancer_type, max_edges=500, layout='spring', figsize=(12, 10)):
        # Determine the appropriate file path based on cancer type
        if cancer_type == 'colorec':
            interaction_file = os.path.join(self.colorec_interaction_dir, "colorec_biogrid_interactions.csv")
            title = "colorec Cancer Gene Interaction Network"
        elif cancer_type == 'panc':
            interaction_file = os.path.join(self.panc_interaction_dir, "panc_biogrid_interactions.csv")
            title = "panc Cancer Gene Interaction Network"
        else:
            raise ValueError("cancer_type must be either 'colorec' or 'panc'")
        
        # Check if the interaction file exists
        if not os.path.exists(interaction_file):
            print(f"Interaction file not found at: {interaction_file}")
            print("You might need to run get_biogrid_interactions() first.")
            return None
        
        # Load the interaction data
        print(f"Loading interaction data from: {interaction_file}")
        interactions_df = pd.read_csv(interaction_file)
        
        # Extract the interaction pairs
        required_columns = [
            'Systematic Name Interactor A', 'Systematic Name Interactor B',
            'Official Symbol Interactor A', 'Official Symbol Interactor B'
        ]
        
        if all(col in interactions_df.columns for col in required_columns):
            # Create a new graph
            G = nx.Graph()
            
            # Create a new DataFrame to hold processed interaction data
            processed_df = interactions_df.copy()
            
            # Replace missing systematic names with official symbols when available
            missing_sysname_a = (
                (processed_df['Systematic Name Interactor A'] == '-') | 
                (processed_df['Systematic Name Interactor A'] == '') | 
                (processed_df['Systematic Name Interactor A'].isna())
            )
            
            missing_sysname_b = (
                (processed_df['Systematic Name Interactor B'] == '-') | 
                (processed_df['Systematic Name Interactor B'] == '') | 
                (processed_df['Systematic Name Interactor B'].isna())
            )
            
            # Replace missing systematic names with official symbols
            processed_df.loc[missing_sysname_a, 'Systematic Name Interactor A'] = (
                processed_df.loc[missing_sysname_a, 'Official Symbol Interactor A']
            )
            
            processed_df.loc[missing_sysname_b, 'Systematic Name Interactor B'] = (
                processed_df.loc[missing_sysname_b, 'Official Symbol Interactor B']
            )
            
            # Now filter out any remaining rows where either name is still missing
            valid_interactions = processed_df[
                (~processed_df['Systematic Name Interactor A'].isna()) &
                (~processed_df['Systematic Name Interactor B'].isna()) &
                (processed_df['Systematic Name Interactor A'] != '') &
                (processed_df['Systematic Name Interactor B'] != '')
            ]
            
            print(f"Replaced {sum(missing_sysname_a)} missing systematic names for interactor A with official symbols")
            print(f"Replaced {sum(missing_sysname_b)} missing systematic names for interactor B with official symbols")
            print(f"Filtered out {len(processed_df) - len(valid_interactions)} interactions with missing identifiers")
            
            interactions_df = valid_interactions
            
            # Limit the number of interactions to prevent visualization overcrowding
            if len(interactions_df) > max_edges:
                print(f"Limiting visualization to {max_edges} interactions (out of {len(interactions_df)} total)")
                interactions_df = interactions_df.sample(max_edges, random_state=42)
            
            # Add edges for each interaction pair
            for _, row in interactions_df.iterrows():
                gene_a = row['Systematic Name Interactor A']
                gene_b = row['Systematic Name Interactor B']
                
                # Add nodes (genes) to the graph
                if gene_a not in G.nodes():
                    G.add_node(gene_a)
                if gene_b not in G.nodes():
                    G.add_node(gene_b)
                
                # Add edge (interaction) between the genes
                G.add_edge(gene_a, gene_b)
            
            # Display basic network statistics
            print(f"Network Statistics:")
            print(f"Number of genes (nodes): {G.number_of_nodes()}")
            print(f"Number of interactions (edges): {G.number_of_edges()}")
            
            # Visualize the graph
            plt.figure(figsize=figsize)
            
            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)  # Increased k value for longer edges and more spacing
            elif layout == 'circular':
                pos = nx.circular_layout(G, scale=1.5)  # Increased scale for more spacing
            elif layout == 'random':
                pos = nx.random_layout(G, seed=42)
            else:
                pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)  # default to spring layout with improved spacing
            
            # Draw the graph with node labels (gene names)
            nx.draw_networkx(
                G, 
                pos=pos,
                node_size=300,  # Increased from 50 to 300 to accommodate labels
                node_color='skyblue',
                edge_color='gray',
                alpha=0.7,
                with_labels=True,  # Show gene names on nodes
                font_size=8,        # Small font size for readability
                font_color='black',
                font_weight='bold'
            )
            
            # Add title and other visual elements
            plt.title(title)
            plt.axis('off')  # Hide axis
            
            # Show the plot
            plt.tight_layout()
            plt.show()
            
            return G
        else:
            print("Required columns not found in the interaction data.")
            print("Available columns:", interactions_df.columns.tolist())
            return None
        
    def _handle_nan_values(self, dataframe):
        """
        Handle missing values and infinities in a dataframe by imputing with column means.
        
        Args:
            dataframe: The pandas DataFrame to process
            
        Returns:
            DataFrame with NaN values and infinities handled
        """
        # Handle missing values by filling with column means (gene averages)
        processed_df = dataframe.fillna(dataframe.mean())
        
        # Replace any remaining infinities with NaN, then fill with column means
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        processed_df = processed_df.fillna(processed_df.mean())
        
        # Verify no NaN or infinities remain
        if processed_df.isna().sum().sum() > 0:
            print("Warning: Some NaN values remain after imputation. Using additional fallback...")
            # Fallback: replace any remaining NaNs with 0
            processed_df = processed_df.fillna(0)
            
        return processed_df
        
    def plot_methylation_heatmap(self, cancer_type, n_top_genes=50, cluster_rows=False, cluster_cols=True, figsize=(16, 10), cmap='coolwarm'):
        """
        Create an informative heatmap of methylation data with clustering and annotations.
        
        Args:
            cancer_type: Either "colorec" or "panc"
            n_top_genes: Number of most variable genes to display (default: 50)
            cluster_rows: Whether to cluster patients (rows) (default: False)
            cluster_cols: Whether to cluster genes (columns) (default: True)
            figsize: Figure size as (width, height) tuple (default: (16, 10))
            cmap: Colormap to use (default: 'coolwarm')
        """
        if cancer_type == "colorec" or cancer_type == "colorec":
            methylation = self.harmonized_colorec["methylation"]
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type == "panc" or cancer_type == "panc":
            methylation = self.harmonized_panc["methylation"]
            clinical = self.harmonized_panc["clinical"]
            title_prefix = "panc"
        else:
            raise ValueError("cancer_type must be either 'colorec'/'colorec' or 'panc'/'panc'")
        
        print(f"Processing methylation data for {title_prefix} cancer...")
        
        # Extract the methylation levels (without patient_id column)
        methylation_levels = methylation.iloc[:, 1:]
        
        # Check for missing values
        missing_percent = (methylation_levels.isna().sum().sum() / 
                          (methylation_levels.shape[0] * methylation_levels.shape[1])) * 100
        print(f"Missing values: {missing_percent:.2f}%")
        
        # Handle missing values using the helper method
        print("Handling missing values...")
        methylation_levels = self._handle_nan_values(methylation_levels)
        
        # Select top N most variable genes to make visualization manageable
        print(f"Selecting top {n_top_genes} most variable genes...")
        gene_variances = methylation_levels.var().sort_values(ascending=False)
        top_genes = gene_variances.index[:n_top_genes].tolist()
        methylation_subset = methylation_levels[top_genes]
        
        # Create a DataFrame for the heatmap with patient IDs as index
        heatmap_data = methylation_subset.copy()
        heatmap_data.index = methylation['patient_id']
        
        # Prepare clinical annotation if available (like tumor stage, survival status)
        row_colors = None
        stage_color_map = None
        
        try:
            print("Generating heatmap with gene clustering only...")
            # Set up the clustering parameters - only cluster genes (columns)
            row_cluster = False  # Don't cluster patients
            col_cluster = True   # Do cluster genes
            
            # Create the heatmap with clustering
            g = sns.clustermap(
                heatmap_data,
                cmap=cmap,
                center=0,
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                row_colors=row_colors,
                xticklabels=True,
                yticklabels=True if len(heatmap_data) <= 30 else 30,  
                figsize=figsize,
                cbar_kws={"label": "Methylation Level"}
            )
            
            # Adjust the main title
            plt.suptitle(f"{title_prefix} Cancer DNA Methylation Pattern\n(Top {n_top_genes} Most Variable Genes - Gene Clustering Only)", 
                         fontsize=16, y=1.02)
            
            # If we have row_colors for tumor stages, add a legend
            if row_colors is not None and stage_color_map is not None:
                # Add legend for tumor stages
                handles = [plt.Rectangle((0,0),1,1, color=color) for color in stage_color_map.values()]
                labels = list(stage_color_map.keys())
                
                legend = plt.legend(handles, labels, title="Tumor Stage", 
                                   loc="center left", bbox_to_anchor=(1, 0.5))
                plt.gca().add_artist(legend)
                
            # Rotate the gene names for better readability
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Make patient IDs readable
            if len(heatmap_data) > 0:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)
            
        except Exception as e:
            print(f"Clustering failed with error: {e}")
            print("Falling back to simple heatmap without clustering...")
            
            # Close previous figure
            plt.close()
            
            # Create a new figure for simple heatmap
            plt.figure(figsize=figsize)
            
            # Create simple heatmap without clustering
            ax = sns.heatmap(
                heatmap_data.iloc[:30, :n_top_genes],  # Limit rows to make it manageable
                cmap=cmap,
                center=0,
                xticklabels=True,
                yticklabels=False,
                cbar_kws={"label": "Methylation Level"}
            )
            
            plt.title(f"{title_prefix} Cancer DNA Methylation Pattern\n(Top {n_top_genes} Most Variable Genes - No Clustering)", 
                      fontsize=16)
            plt.xlabel("Genes")
            plt.ylabel("Patients")
            
            # Rotate the gene names for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
        plt.tight_layout()
        print("Displaying plot...")
        plt.show()
        
        return

    def plot_rnaseq_heatmap(self, cancer_type, n_top_genes=50, cluster_rows=False, cluster_cols=True, figsize=(16, 10), cmap='viridis', log_transform=True):
        """
        Create an informative heatmap of gene-level mRNA expression from RNA-seq data with clustering and annotations.
        
        Args:
            cancer_type: Either "colorec" or "panc"
            n_top_genes: Number of most variable genes to display (default: 50)
            cluster_rows: Whether to cluster patients (rows) (default: False)
            cluster_cols: Whether to cluster genes (columns) (default: True)
            figsize: Figure size as (width, height) tuple (default: (16, 10))
            cmap: Colormap to use (default: 'viridis')
            log_transform: Whether to apply log2 transformation to the expression data (default: True)
        """
        if cancer_type == "colorec" or cancer_type == "colorec":
            rnaseq = self.harmonized_colorec["rnaseq"]
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type == "panc" or cancer_type == "panc":
            rnaseq = self.harmonized_panc["rnaseq"]
            clinical = self.harmonized_panc["clinical"]
            title_prefix = "panc"
        else:
            raise ValueError("cancer_type must be either 'colorec'/'colorec' or 'panc'/'panc'")
        
        print(f"Processing RNA-seq data for {title_prefix} cancer...")
        
        # Extract the gene expression levels (without patient_id column)
        expression_levels = rnaseq.iloc[:, 1:]
        
        # Check for missing values
        missing_percent = (expression_levels.isna().sum().sum() / 
                          (expression_levels.shape[0] * expression_levels.shape[1])) * 100
        print(f"Missing values: {missing_percent:.2f}%")
        
        # Handle missing values using the helper method
        print("Handling missing values...")
        expression_levels = self._handle_nan_values(expression_levels)
        
        # Apply log2 transformation if needed (common for RNA-seq data)
        if log_transform:
            print("Applying log2 transformation to RNA-seq data...")
            # Add a small value to avoid log(0)
            expression_levels = np.log2(expression_levels + 1)
        
        # Select top N most variable genes to make visualization manageable
        print(f"Selecting top {n_top_genes} most variable genes...")
        gene_variances = expression_levels.var().sort_values(ascending=False)
        top_genes = gene_variances.index[:n_top_genes].tolist()
        expression_subset = expression_levels[top_genes]
        
        # Create a DataFrame for the heatmap with patient IDs as index
        heatmap_data = expression_subset.copy()
        heatmap_data.index = rnaseq['patient_id']
        
        # Prepare clinical annotation if available (like tumor stage, survival status)
        row_colors = None
        stage_color_map = None
        
        # Try to add clinical annotations if the data is available
        if 'stage' in clinical.columns:
            print("Adding tumor stage annotations to heatmap...")
            # Get patients in the same order as the heatmap data
            patient_stages = clinical.set_index('patient_id').loc[heatmap_data.index, 'stage']
            
            # Create color mapping for tumor stages
            unique_stages = patient_stages.unique()
            colors = sns.color_palette("Set2", len(unique_stages))
            stage_color_map = dict(zip(unique_stages, colors))
            
            # Create row colors for the heatmap
            row_colors = patient_stages.map(stage_color_map)
        
        try:
            print("Generating heatmap with clustering...")
            # Set up the clustering parameters
            row_cluster = cluster_rows
            col_cluster = cluster_cols
            
            # Create the heatmap with clustering
            g = sns.clustermap(
                heatmap_data,
                cmap=cmap,
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                row_colors=row_colors,
                xticklabels=True,
                yticklabels=True if len(heatmap_data) <= 30 else 30,  
                figsize=figsize,
                cbar_kws={"label": "log2(Expression + 1)" if log_transform else "Expression Level"}
            )
            
            # Adjust the main title
            clustering_info = []
            if row_cluster:
                clustering_info.append("Patient Clustering")
            if col_cluster:
                clustering_info.append("Gene Clustering")
            
            clustering_text = " & ".join(clustering_info) if clustering_info else "No Clustering"
            
            plt.suptitle(f"{title_prefix} Cancer mRNA Expression Pattern\n(Top {n_top_genes} Most Variable Genes - {clustering_text})", 
                         fontsize=16, y=1.02)
            
            # If we have row_colors for tumor stages, add a legend
            if row_colors is not None and stage_color_map is not None:
                # Add legend for tumor stages
                handles = [plt.Rectangle((0,0),1,1, color=color) for color in stage_color_map.values()]
                labels = list(stage_color_map.keys())
                
                legend = plt.legend(handles, labels, title="Tumor Stage", 
                                   loc="center left", bbox_to_anchor=(1, 0.5))
                plt.gca().add_artist(legend)
                
            # Rotate the gene names for better readability
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Make patient IDs readable
            if len(heatmap_data) > 0:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)
            
        except Exception as e:
            print(f"Clustering failed with error: {e}")
            print("Falling back to simple heatmap without clustering...")
            
            # Close previous figure
            plt.close()
            
            # Create a new figure for simple heatmap
            plt.figure(figsize=figsize)
            
            # Create simple heatmap without clustering
            ax = sns.heatmap(
                heatmap_data.iloc[:30, :n_top_genes],  # Limit rows to make it manageable
                cmap=cmap,
                xticklabels=True,
                yticklabels=False,
                cbar_kws={"label": "log2(Expression + 1)" if log_transform else "Expression Level"}
            )
            
            plt.title(f"{title_prefix} Cancer mRNA Expression Pattern\n(Top {n_top_genes} Most Variable Genes - No Clustering)", 
                      fontsize=16)
            plt.xlabel("Genes")
            plt.ylabel("Patients")
            
            # Rotate the gene names for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
        plt.tight_layout()
        print("Displaying plot...")
        plt.show()
        
        return

    def plot_omics_heatmap(self, cancer_type, data_type='rnaseq', n_top_genes=50, cluster_rows=False, cluster_cols=True, 
                         figsize=(16, 10), cmap='coolwarm', log_transform=True):
        """
        Create an informative heatmap of different omics data types with clustering and annotations.
        
        Args:
            cancer_type: Either "colorec"/"colorec" or "panc"/"panc"
            data_type: Type of omics data to visualize ('methylation', 'rnaseq', or 'scnv')
            n_top_genes: Number of most variable genes to display (default: 50)
            cluster_rows: Whether to cluster patients (rows) (default: False)
            cluster_cols: Whether to cluster genes (columns) (default: True)
            figsize: Figure size as (width, height) tuple (default: (16, 10))
            cmap: Colormap to use (default: None, will be set based on data_type)
            log_transform: Whether to apply log2 transformation to the expression data (default: True, only applies to rnaseq)
        """
        # Validate cancer type
        if cancer_type in ["colorec", "colorec"]:
            omics_data = self.harmonized_colorec
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type in ["panc", "panc"]:
            omics_data = self.harmonized_panc
            clinical = self.harmonized_panc["clinical"]
            title_prefix = "panc"
        else:
            raise ValueError("cancer_type must be either 'colorec'/'colorec' or 'panc'/'panc'")
        
        # Validate data type and set appropriate defaults
        if data_type == 'methylation':
            data = omics_data["methylation"]
            data_label = "Methylation Level"
            title_data_type = "DNA Methylation"
            if cmap is None:
                cmap = 'coolwarm'
            center = 0  # for diverging colormap
            apply_log = False  # Methylation data doesn't need log transformation
        elif data_type == 'rnaseq':
            data = omics_data["rnaseq"]
            data_label = "log2(Expression + 1)" if log_transform else "Expression Level"
            title_data_type = "mRNA Expression"
            if cmap is None:
                cmap = 'viridis'
            center = None  # for sequential colormap
            apply_log = log_transform
        elif data_type == 'scnv':
            data = omics_data["scnv"]
            data_label = "Copy Number Variation"
            title_data_type = "Somatic Copy Number Variation"
            if cmap is None:
                cmap = 'RdBu_r'  # Red-Blue reversed is good for CNV (red for amplification, blue for deletion)
            center = 0  # CNV data is centered around 0
            apply_log = False  # CNV data doesn't need log transformation
        else:
            raise ValueError("data_type must be one of 'methylation', 'rnaseq', or 'scnv'")
        
        print(f"Processing {title_data_type} data for {title_prefix} cancer...")
        
        # Extract the data values (without patient_id column)
        values = data.iloc[:, 1:]
        
        # Check for missing values
        missing_percent = (values.isna().sum().sum() / 
                          (values.shape[0] * values.shape[1])) * 100
        print(f"Missing values: {missing_percent:.2f}%")
        
        # Handle missing values using the helper method
        print("Handling missing values...")
        values = self._handle_nan_values(values)
        
        # Apply log2 transformation if needed (common for RNA-seq data)
        if apply_log:
            print(f"Applying log2 transformation to {data_type} data...")
            # Add a small value to avoid log(0)
            values = np.log2(values + 1)
        
        # Select top N most variable genes to make visualization manageable
        print(f"Selecting top {n_top_genes} most variable genes...")
        gene_variances = values.var().sort_values(ascending=False)
        top_genes = gene_variances.index[:n_top_genes].tolist()
        data_subset = values[top_genes]
        
        # Create a DataFrame for the heatmap with patient IDs as index
        heatmap_data = data_subset.copy()
        heatmap_data.index = data['patient_id']
        
        # Prepare clinical annotation if available (like tumor stage, survival status)
        row_colors = None
        stage_color_map = None
        

        
        try:
            print("Generating heatmap with clustering...")
            # Set up the clustering parameters
            row_cluster = cluster_rows
            col_cluster = cluster_cols
            
            # Create the heatmap with clustering
            g = sns.clustermap(
                heatmap_data,
                cmap=cmap,
                center=center,
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                row_colors=row_colors,
                xticklabels=True,
                yticklabels=True if len(heatmap_data) <= 30 else 30,  
                figsize=figsize,
                cbar_kws={"label": data_label}
            )
            
            # Adjust the main title
            clustering_info = []
            if row_cluster:
                clustering_info.append("Patient Clustering")
            if col_cluster:
                clustering_info.append("Gene Clustering")
            
            clustering_text = " & ".join(clustering_info) if clustering_info else "No Clustering"
            
            plt.suptitle(f"{title_prefix} Cancer {title_data_type} Pattern\n(Top {n_top_genes} Most Variable Genes - {clustering_text})", 
                         fontsize=16, y=1.02)
            
            # If we have row_colors for tumor stages, add a legend
            if row_colors is not None and stage_color_map is not None:
                # Add legend for tumor stages
                handles = [plt.Rectangle((0,0),1,1, color=color) for color in stage_color_map.values()]
                labels = list(stage_color_map.keys())
                
                legend = plt.legend(handles, labels, title="Tumor Stage", 
                                   loc="center left", bbox_to_anchor=(1, 0.5))
                plt.gca().add_artist(legend)
            
            # Improve x-axis label legibility
            # Rotate labels, increase font size, and adjust position for better readability
            plt.setp(g.ax_heatmap.get_xticklabels(), 
                    rotation=45,                # 45 degree rotation is more readable than 90
                    ha='right',                 # Horizontal alignment right
                    fontsize=12,                # Larger font size (was 8)
                    fontweight='bold',          # Make labels bold
                    rotation_mode='anchor')     # Anchor rotation at right side of text
            
            # Add more space at the bottom to prevent label cutoff
            plt.subplots_adjust(bottom=0.2)
            
            # Make patient IDs readable
            if len(heatmap_data) > 0:
                plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)
            
            # Add explicit axis labels to clustermap
            g.ax_heatmap.set_xlabel("Genes", fontsize=14, fontweight='bold', labelpad=10)
            g.ax_heatmap.set_ylabel("Patients", fontsize=14, fontweight='bold', labelpad=10)
        except Exception as e:
            print(f"Clustering failed with error: {e}")
            print("Falling back to simple heatmap without clustering...")
            
            # Close previous figure
            plt.close()
            
            # Create a new figure for simple heatmap
            plt.figure(figsize=figsize)
            
            # Create simple heatmap without clustering
            ax = sns.heatmap(
                heatmap_data.iloc[:30, :n_top_genes],  # Limit rows to make it manageable
                cmap=cmap,
                center=center,
                xticklabels=True,
                yticklabels=False,
                cbar_kws={"label": data_label}
            )
            
            plt.title(f"{title_prefix} Cancer {title_data_type} Pattern\n(Top {n_top_genes} Most Variable Genes - No Clustering)", 
                      fontsize=16)
            plt.xlabel("Genes", fontsize=14, fontweight='bold')
            plt.ylabel("Patients", fontsize=14, fontweight='bold')
            
            # Improve x-axis label legibility for the fallback heatmap
            plt.setp(ax.get_xticklabels(), 
                    rotation=45,                # 45 degree angle
                    ha='right',                 # Horizontal alignment
                    fontsize=12,                # Larger font size
                    fontweight='bold',          # Make labels bold
                    rotation_mode='anchor')     # Anchor rotation at right side
            
            # Add more space at the bottom to prevent label cutoff
            plt.subplots_adjust(bottom=0.2)
            
        plt.tight_layout()
        print("Displaying plot...")
        plt.show()
        
        return

    def plot_omics_by_stage(self, cancer_type, data_type='rnaseq', n_genes=5, 
                            figsize=(20, 12), palette='viridis', select_by='stage_variance'):
        """
        Create violin plots showing distribution of omics features across cancer stages.
        
        Args:
            cancer_type: "colorec"/"colorec" or "panc"/"panc"
            data_type: 'methylation', 'rnaseq', or 'scnv'
            n_genes: Number of top variable genes to display (default: 5)
            figsize: Figure size tuple (default: (20, 12))
            palette: Color palette for stages (default: 'viridis')
            select_by: Method to select top genes ('overall_variance' or 'stage_variance')
        """
        # Data loading and validation
        if cancer_type in ["colorec", "colorec"]:
            data = self.harmonized_colorec[data_type]
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type in ["panc", "panc"]:
            data = self.harmonized_panc[data_type]
            clinical = self.harmonized_panc["clinical"]
            title_prefix = "panc"
        else:
            raise ValueError("Invalid cancer_type")

        # Check for stage column - look for 'stage' in column names if 'pathologic_stage' doesn't exist
        if 'pathologic_stage' not in clinical.columns:
            stage_columns = [col for col in clinical.columns if 'stage' in col.lower()]
            if not stage_columns:
                raise ValueError(f"Clinical data missing stage information. Available columns: {clinical.columns.tolist()}")
            stage_column = stage_columns[0]
            print(f"Using '{stage_column}' as the stage column")
        else:
            stage_column = 'pathologic_stage'

        # Merge omics data with clinical stages
        merged = data.merge(clinical[['patient_id', stage_column]], on='patient_id')
        
        # Print stage counts to help diagnose issues
        stage_counts = merged[stage_column].value_counts().sort_index()
        print(f"Patient counts per stage in {cancer_type} cancer:")
        for stage, count in stage_counts.items():
            print(f"  {stage}: {count} patients")
        
        # Get data values excluding patient_id and stage
        values = merged.iloc[:, 1:-1]  # Exclude patient_id and stage
        
        # Select top genes based on specified method
        if select_by == 'overall_variance':
            # Original method: select genes with highest overall variance
            print(f"Selecting top {n_genes} genes by overall variance...")
            top_genes = values.var().sort_values(ascending=False).head(n_genes).index.tolist()
        elif select_by == 'stage_variance':
            # New method: select genes with highest variance between stage means
            print(f"Selecting top {n_genes} genes by variance across stages...")
            
            # Calculate mean value for each gene per stage
            stage_means = pd.DataFrame(index=values.columns)
            for stage in stage_counts.index:
                # Get indices of samples in current stage
                stage_samples = merged[merged[stage_column] == stage].index
                if len(stage_samples) > 0:
                    # Calculate mean for each gene in this stage
                    stage_means[stage] = values.iloc[stage_samples].mean()
            
            # Calculate variance of means across stages for each gene
            # This identifies genes whose average expression differs most between stages
            stage_variances = stage_means.var(axis=1).sort_values(ascending=False)
            top_genes = stage_variances.head(n_genes).index.tolist()
            
            # Print top genes and their variance across stages
            print("Top genes by variance across stages:")
            for gene in top_genes:
                means_str = ", ".join([f"{stage}: {stage_means.loc[gene, stage]:.4f}" for stage in stage_means.columns])
                print(f"  {gene} - Variance: {stage_variances[gene]:.4f} - Means: {means_str}")
        else:
            raise ValueError("select_by must be either 'overall_variance' or 'stage_variance'")
            
        print(f"Top {n_genes} genes selected: {', '.join(top_genes)}")
        
        # Melt data for plotting
        plot_data = merged.melt(
            id_vars=['patient_id', stage_column],
            value_vars=top_genes,
            var_name='gene',
            value_name='value'
        )

        # Use default stage order or determine from data
        if all(stage.lower() in ['stage i', 'stage ii', 'stage iii', 'stage iv', 'stagei', 'stageii', 'stageiii', 'stageiv'] 
              for stage in stage_counts.index):
            # Convert to consistent format if needed
            stage_order = ['stagei', 'stageii', 'stageiii', 'stageiv']
            # Filter to only include stages that exist in the data
            stage_order = [stage for stage in stage_order if stage in stage_counts.index]
        else:
            # If stages don't follow expected pattern, use alphabetical order
            stage_order = sorted(stage_counts.index)
        
        # Create subplot for each gene
        plt.figure(figsize=figsize)
        for i, gene in enumerate(top_genes):
            plt.subplot(n_genes, 1, i+1)
            
            # Get data for this gene
            gene_data = plot_data[plot_data['gene'] == gene]
            
            # Create violin plot to show the distribution
            sns.violinplot(
                x=stage_column, 
                y='value', 
                data=gene_data,
                order=stage_order,
                palette=palette
            )
            
            # Add individual points
            sns.stripplot(
                x=stage_column, 
                y='value', 
                data=gene_data, 
                order=stage_order,
                size=4, 
                color='black',
                alpha=0.4,
                jitter=True
            )
            
            plt.title(f"{gene}")
            plt.ylabel("Expression Value" if data_type == 'rnaseq' else 
                      "Methylation Value" if data_type == 'methylation' else "CNV Value")
            
            # Get data range for this gene to calculate proportional spacing
            y_min = gene_data['value'].min()
            y_max = gene_data['value'].max()
            y_range = y_max - y_min
            
            # Add statistics with dynamic spacing based on data range
            for j, stage in enumerate(stage_order):
                stage_values = gene_data[gene_data[stage_column] == stage]['value']
                if len(stage_values) > 0:
                    # Calculate base position above the maximum data point
                    y_pos = stage_values.max() + (y_range * 0.05)  # Start 5% of range above max
                    
                    # Calculate statistics
                    median = stage_values.median()
                    q1 = stage_values.quantile(0.25)
                    q3 = stage_values.quantile(0.75)
                    iqr = q3 - q1
                    
                    # Use proportional spacing based on data range (15% of range between items)
                    spacing_factor = y_range * 0.15
                    
                    # Display sample size with white background for better visibility
                    plt.text(j, y_pos, f"n={len(stage_values)}", 
                            horizontalalignment='center', size='small', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    
                    # Display median with increased spacing and background
                    plt.text(j, y_pos + spacing_factor, f"Median: {median:.2f}", 
                            horizontalalignment='center', size='small', color='blue', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    
                    # Display IQR with further increased spacing and background
                    plt.text(j, y_pos + (spacing_factor * 2), f"IQR: {iqr:.2f}", 
                            horizontalalignment='center', size='small', color='blue', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # After adding all annotations, adjust y-axis limits for the entire subplot
            ax = plt.gca()
            y_lim = ax.get_ylim()
            # Allow 40% more space above the data for annotations
            ax.set_ylim(y_lim[0], y_max + y_range * 0.4)
        
        selection_method = "Genes with Highest Variance Between Stages" if select_by == 'stage_variance' else "Most Variable Genes Overall"
        plt.suptitle(f"{title_prefix} Cancer {data_type.upper()} Patterns by Pathologic Stage\n({selection_method})", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_omics_survival(self, cancer_type, data_type='rnaseq', genes=None, n_top_genes=5, 
                            figsize=(18, 12), add_trendline=True, log_transform=True,
                            bubble_size_by=None, layout='grid'):
        """
        Create scatter/bubble plots showing the relationship between gene omics values and survival time.
        
        Args:
            cancer_type: "colorec"/"colorec" or "panc"/"panc"
            data_type: Type of omics data ('methylation', 'rnaseq', or 'scnv')
            genes: List of specific genes to plot. If None, top variable genes will be selected.
            n_top_genes: Number of top variable genes to display if genes=None (default: 5)
            figsize: Figure size tuple (default: (18, 12))
            add_trendline: Whether to add a trend line to each plot (default: True)
            log_transform: Whether to log2 transform expression data (default: True, only for rnaseq)
            bubble_size_by: Optional column name to determine bubble size (e.g., 'age')
            layout: Plot layout ('grid' or 'individual')
        
        Returns:
            Matplotlib figure with the generated plots
        """
        # Determine which dataset to use based on cancer type
        if cancer_type in ["colorec", "colorec"]:
            omics_data = self.harmonized_colorec[data_type]
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type in ["panc", "panc"]:
            omics_data = self.harmonized_panc[data_type]
            clinical = self.harmonized_panc["clinical"] 
            title_prefix = "panc"
        else:
            raise ValueError("cancer_type must be either 'colorec'/'colorec' or 'panc'/'panc'")
            
        # Check if necessary survival columns exist in clinical data
        required_fields = ['status', 'overall_survival']
        missing_fields = [field for field in required_fields if field not in clinical.columns]
        
        if missing_fields:
            raise ValueError(f"Clinical data missing required survival fields: {', '.join(missing_fields)}. "
                            f"Available columns: {clinical.columns.tolist()}")
        
        # Merge omics data with clinical data
        print(f"Merging {data_type} data with clinical survival data...")
        merged_df = omics_data.merge(clinical[['patient_id', 'status', 'overall_survival']], 
                                    on='patient_id', how='inner')
        
        print(f"Total patients with complete data: {len(merged_df)}")
        print(f"Status distribution: {merged_df['status'].value_counts().to_dict()}")
        
        # If bubble_size_by is provided, make sure it exists in clinical data
        if bubble_size_by is not None and bubble_size_by not in clinical.columns:
            print(f"Warning: {bubble_size_by} not found in clinical data. Using fixed bubble size instead.")
            bubble_size_by = None
        
        # Handle bubble size column if provided
        if bubble_size_by is not None:
            print(f"Using {bubble_size_by} for bubble sizes.")
            merged_df = merged_df.merge(clinical[['patient_id', bubble_size_by]], 
                                      on='patient_id', how='inner')
        
        # Set data type specific settings
        if data_type == 'rnaseq':
            data_label = "Gene Expression"
            if log_transform:
                # Extract gene columns (all columns except patient_id, status, overall_survival, and optional bubble_size_by)
                gene_cols = [col for col in merged_df.columns 
                           if col not in ['patient_id', 'status', 'overall_survival', bubble_size_by]]
                
                # Apply log2 transformation to gene expression values
                print("Applying log2 transformation to gene expression data...")
                for col in gene_cols:
                    merged_df[col] = np.log2(merged_df[col] + 1)  # Add 1 to avoid log(0)
                
                data_label = "log2(Gene Expression + 1)"
                
        elif data_type == 'methylation':
            data_label = "DNA Methylation"
        elif data_type == 'scnv':
            data_label = "Copy Number Variation"
        else:
            raise ValueError("data_type must be one of 'methylation', 'rnaseq', or 'scnv'")
        
        # If no specific genes provided, select top variable genes
        if genes is None:
            print(f"Selecting top {n_top_genes} most variable genes...")
            # Get all gene columns (exclude patient_id, status, overall_survival, and bubble_size_by)
            gene_cols = [col for col in merged_df.columns 
                       if col not in ['patient_id', 'status', 'overall_survival', bubble_size_by]]
            
            # Calculate variance for each gene
            gene_var = merged_df[gene_cols].var().sort_values(ascending=False)
            genes = gene_var.index[:n_top_genes].tolist()
            
            print(f"Selected genes: {', '.join(genes)}")
        else:
            # Validate all specified genes exist in the dataset
            missing_genes = [gene for gene in genes if gene not in merged_df.columns]
            if missing_genes:
                raise ValueError(f"The following genes were not found in the {data_type} dataset: {', '.join(missing_genes)}")
        
        # Create the figure with subplots
        n_genes = len(genes)
        
        if layout == 'grid':
            # Calculate grid dimensions based on number of genes
            n_cols = min(3, n_genes)  # Maximum 3 columns
            n_rows = (n_genes + n_cols - 1) // n_cols  # Ceiling division for number of rows
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            # Flatten axes array for easier indexing if it's a multi-dimensional array
            if n_genes > 1:
                axes = axes.flatten()
        else:  # individual layout
            fig = plt.figure(figsize=figsize)
            axes = [plt.subplot(n_genes, 1, i+1) for i in range(n_genes)]
        
        # Create a scatter plot for each gene
        for i, gene in enumerate(genes):
            ax = axes[i] if i < len(axes) else axes[-1]  # Use the last axis if we have more genes than plots
            
            # Get x and y values for the scatter plot
            x = merged_df[gene]
            y = merged_df['overall_survival']
            status = merged_df['status']
            
            # Create color mapping for status (living vs deceased)
            # 0: living/censored (blue), 1: deceased (red)
            colors = ['#3274A1' if s == 0 else '#E1484E' for s in status]
            
            # Determine bubble sizes if applicable
            if bubble_size_by is not None:
                # Normalize the bubble size variable to get reasonable bubble sizes
                size_values = merged_df[bubble_size_by]
                # Handle potential NaN values
                size_values = size_values.fillna(size_values.median())
                
                # Scale to reasonable bubble sizes (between 20 and 200)
                min_size, max_size = 20, 200
                if size_values.min() != size_values.max():  # Avoid division by zero
                    normalized_sizes = min_size + (size_values - size_values.min()) * (max_size - min_size) / (size_values.max() - size_values.min())
                else:
                    normalized_sizes = np.ones_like(size_values) * ((min_size + max_size) / 2)
                
                # Create the scatter plot with variable bubble sizes
                scatter = ax.scatter(x, y, c=colors, s=normalized_sizes, alpha=0.6, edgecolors='white', linewidth=0.5)
            else:
                # Create the scatter plot with fixed bubble size
                scatter = ax.scatter(x, y, c=colors, s=80, alpha=0.6, edgecolors='white', linewidth=0.5)
            
            # Add a trend line using polynomial regression if requested
            if add_trendline:
                # Remove NaN values for trend line calculation
                mask = ~(np.isnan(x) | np.isnan(y))
                x_valid = x[mask]
                y_valid = y[mask]
                
                if len(x_valid) > 1:  # Need at least 2 points for a trend line
                    try:
                        # Fit a polynomial of degree 1 (linear) or 2 (quadratic) based on data
                        degree = 1 if len(x_valid) < 20 else 2  # Use quadratic only if we have enough data
                        coeffs = np.polyfit(x_valid, y_valid, degree)
                        poly = np.poly1d(coeffs)
                        
                        # Generate x values for the trend line
                        x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
                        y_trend = poly(x_trend)
                        
                        # Plot the trend line
                        ax.plot(x_trend, y_trend, color='black', linestyle='--', linewidth=2, 
                               label=f"Trend (degree={degree})")
                        
                        # Add R-squared value to the plot
                        if len(x_valid) > 2:  # Need at least 3 points for meaningful R²
                            y_pred = poly(x_valid)
                            ss_total = np.sum((y_valid - np.mean(y_valid))**2)
                            ss_residual = np.sum((y_valid - y_pred)**2)
                            
                            if ss_total != 0:  # Avoid division by zero
                                r_squared = 1 - (ss_residual / ss_total)
                                # Add R² text to the plot
                                ax.text(0.05, 0.95, f"R² = {r_squared:.3f}", transform=ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    except Exception as e:
                        print(f"Warning: Could not add trend line for {gene}: {str(e)}")
            
            # Add labels and title for each subplot
            ax.set_xlabel(f"{gene} {data_label}", fontsize=12)
            ax.set_ylabel("Overall Survival (Days)", fontsize=12)
            ax.set_title(f"{gene}", fontsize=14)
            
            # Add a grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add correlation coefficient
            try:
                corr = merged_df[[gene, 'overall_survival']].corr().iloc[0, 1]
                ax.text(0.05, 0.89, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {gene}: {str(e)}")
        
        # Add a legend for the status colors
        if n_genes > 0:
            # Create custom legend elements
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#3274A1', markersize=10, label='Living/Censored'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#E1484E', markersize=10, label='Deceased')
            ]
            
            # Add bubble size legend if applicable
            if bubble_size_by is not None:
                # Create legend for bubble sizes
                size_legend = []
                for size, label in zip([min_size, (min_size + max_size) // 2, max_size], 
                                      ['Low', 'Medium', 'High']):
                    size_legend.append(Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor='gray', markersize=np.sqrt(size/5),
                                             label=f"{label} {bubble_size_by}"))
                
                legend_elements.extend(size_legend)
            
            # Place the legend outside the subplots for a grid layout
            if layout == 'grid':
                fig.legend(handles=legend_elements, loc='upper center', 
                          bbox_to_anchor=(0.5, 0), ncol=len(legend_elements))
            else:  # For individual layout, add legend to the last subplot
                axes[-1].legend(handles=legend_elements, loc='lower center', 
                              bbox_to_anchor=(0.5, -0.3), ncol=len(legend_elements))
        
        # Add a main title for the figure
        fig.suptitle(f"{title_prefix} Cancer: Relationship Between {data_type.upper()} and Overall Survival",
                    fontsize=18, y=0.98)
        
        # Remove any unused subplots
        if layout == 'grid':
            for j in range(n_genes, len(axes)):
                fig.delaxes(axes[j])
        
        # Adjust layout
        fig.tight_layout()
        if layout == 'grid':
            plt.subplots_adjust(top=0.9, bottom=0.12)  # Make room for the title and legend
        else:
            plt.subplots_adjust(top=0.95, hspace=0.4)  # Adjust spacing between individual plots
        
        print("Displaying plot...")
        plt.show()
        
        return fig

    def plot_kaplan_meier_by_gene(self, cancer_type, data_type='rnaseq', genes=None, n_top_genes=5, 
                                  split_method='median', log_transform=True, time_unit='days',
                                  figsize=(18, 12), layout='grid', at_risk_table=True):
        """
        Create Kaplan-Meier survival curves stratified by gene values (expression, methylation, or CNV).
        
        Args:
            cancer_type: "colorec"/"colorec" or "panc"/"panc"
            data_type: Type of omics data ('methylation', 'rnaseq', or 'scnv')
            genes: List of specific genes to plot. If None, top variable genes will be selected.
            n_top_genes: Number of top variable genes to display if genes=None (default: 5)
            split_method: Method to stratify patients ('median', 'quartile', 'optimal')
            log_transform: Whether to log2 transform expression data (default: True, only for rnaseq)
            time_unit: Time unit for x-axis (e.g., 'days', 'months', 'years')
            figsize: Figure size tuple (default: (18, 12))
            layout: Plot layout ('grid' or 'individual')
            at_risk_table: Whether to include a table showing patients at risk over time (default: True)
        
        Returns:
            Matplotlib figure with the generated Kaplan-Meier plots
        """
        # Import lifelines package for survival analysis
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test, multivariate_logrank_test
            from lifelines.utils import median_survival_times
        except ImportError:
            raise ImportError("This function requires the lifelines package. Please install it using 'pip install lifelines'.")
        
        # Determine which dataset to use based on cancer type
        if cancer_type in ["colorec", "colorec"]:
            omics_data = self.harmonized_colorec[data_type]
            clinical = self.harmonized_colorec["clinical"]
            title_prefix = "colorec"
        elif cancer_type in ["panc", "panc"]:
            omics_data = self.harmonized_panc[data_type]
            clinical = self.harmonized_panc["clinical"] 
            title_prefix = "panc"
        else:
            raise ValueError("cancer_type must be either 'colorec'/'colorec' or 'panc'/'panc'")
            
        # Check if necessary survival columns exist in clinical data
        required_fields = ['status', 'overall_survival']
        missing_fields = [field for field in required_fields if field not in clinical.columns]
        
        if missing_fields:
            raise ValueError(f"Clinical data missing required survival fields: {', '.join(missing_fields)}. "
                            f"Available columns: {clinical.columns.tolist()}")
        
        # Merge omics data with clinical data
        print(f"Merging {data_type} data with clinical survival data...")
        merged_df = omics_data.merge(clinical[['patient_id', 'status', 'overall_survival']], 
                                     on='patient_id', how='inner')
        
        print(f"Total patients with complete data: {len(merged_df)}")
        status_counts = merged_df['status'].value_counts().to_dict()
        print(f"Status distribution: {status_counts}")
        
        # Handle missing values in the merged dataset
        print("Handling missing values...")
        
        # Get all columns except patient_id for handling NaN values
        data_cols = [col for col in merged_df.columns if col != 'patient_id']
        merged_df[data_cols] = self._handle_nan_values(merged_df[data_cols])
        
        # Set data type specific settings
        if data_type == 'rnaseq':
            data_label = "Expression"
            if log_transform:
                # Extract gene columns (all columns except patient_id, status, overall_survival)
                gene_cols = [col for col in merged_df.columns 
                            if col not in ['patient_id', 'status', 'overall_survival']]
                
                # Apply log2 transformation to gene expression values
                print("Applying log2 transformation to gene expression data...")
                for col in gene_cols:
                    merged_df[col] = np.log2(merged_df[col] + 1)  # Add 1 to avoid log(0)
                
                data_label = "log2(Expression + 1)"
        elif data_type == 'methylation':
            data_label = "Methylation"
        elif data_type == 'scnv':
            data_label = "Copy Number"
        else:
            raise ValueError("data_type must be one of 'methylation', 'rnaseq', or 'scnv'")
        
        # If no specific genes provided, select top variable genes
        if genes is None:
            print(f"Selecting top {n_top_genes} most variable genes...")
            # Get all gene columns (exclude patient_id, status, overall_survival)
            gene_cols = [col for col in merged_df.columns 
                        if col not in ['patient_id', 'status', 'overall_survival']]
            
            # Calculate variance for each gene
            gene_var = merged_df[gene_cols].var().sort_values(ascending=False)
            genes = gene_var.index[:n_top_genes].tolist()
            
            print(f"Selected genes: {', '.join(genes)}")
        else:
            # Validate all specified genes exist in the dataset
            missing_genes = [gene for gene in genes if gene not in merged_df.columns]
            if missing_genes:
                raise ValueError(f"The following genes were not found in the {data_type} dataset: {', '.join(missing_genes)}")
        
        # Create the figure with subplots
        n_genes = len(genes)
        
        # Adjust figure size and layout to accommodate at-risk tables if needed
        if at_risk_table:
            # Make the figure taller to accommodate the at-risk tables
            figsize = (figsize[0], figsize[1] * 1.3)
        
        if layout == 'grid':
            # Calculate grid dimensions based on number of genes
            n_cols = min(3, n_genes)  # Maximum 3 columns
            n_rows = (n_genes + n_cols - 1) // n_cols  # Ceiling division for number of rows
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            # Flatten axes array for easier indexing if it's a multi-dimensional array
            if n_genes > 1:
                axes = axes.flatten()
            else:
                axes = [axes]  # Convert to list for consistent indexing
        else:  # individual layout
            fig = plt.figure(figsize=figsize)
            axes = [plt.subplot(n_genes, 1, i+1) for i in range(n_genes)]
        
        # Store all created axes for later adjustments
        all_axes = []
        
        # Colors for different groups
        group_colors = ['#3274A1', '#E1484E', '#72B566', '#AA6F73', '#6B8E23', '#A52A2A']
        
        # Create a Kaplan-Meier plot for each gene
        for i, gene in enumerate(genes):
            ax = axes[i] if i < len(axes) else axes[-1]  # Use the last axis if we have more genes than plots
            all_axes.append(ax)
            
            gene_values = merged_df[gene]
            durations = merged_df['overall_survival']
            events = merged_df['status']
            
            # Stratify patients based on gene values
            if split_method == 'median':
                # Split patients into two groups by median value
                threshold = gene_values.median()
                high_mask = gene_values >= threshold
                low_mask = gene_values < threshold
                
                groups = [
                    ('High', high_mask, group_colors[0]),
                    ('Low', low_mask, group_colors[1])
                ]
                
            elif split_method == 'quartile':
                # Split patients into quartiles
                q1 = gene_values.quantile(0.25)
                q3 = gene_values.quantile(0.75)
                
                low_mask = gene_values <= q1
                high_mask = gene_values >= q3
                
                groups = [
                    ('High (>Q3)', high_mask, group_colors[0]),
                    ('Low (<Q1)', low_mask, group_colors[1])
                ]
                
            elif split_method == 'tertile':
                # Split patients into tertiles
                t1 = gene_values.quantile(1/3)
                t2 = gene_values.quantile(2/3)
                
                low_mask = gene_values <= t1
                mid_mask = (gene_values > t1) & (gene_values < t2)
                high_mask = gene_values >= t2
                
                groups = [
                    ('High', high_mask, group_colors[0]),
                    ('Medium', mid_mask, group_colors[2]),
                    ('Low', low_mask, group_colors[1])
                ]
                
            elif split_method == 'optimal':
                # Find the threshold that maximizes the log-rank test statistic
                # This is a simple implementation; more advanced methods exist
                unique_values = np.unique(gene_values)
                
                if len(unique_values) <= 1:
                    print(f"Warning: Gene {gene} has only one unique value. Skipping optimal split calculation.")
                    threshold = gene_values.median()
                else:
                    # Search for optimal cutpoint by testing percentiles
                    best_pvalue = 1.0
                    best_threshold = gene_values.median()
                    
                    # Test percentiles from 10% to 90% in steps of 5%
                    for percentile in range(10, 91, 5):
                        test_threshold = gene_values.quantile(percentile / 100)
                        
                        # Skip if threshold creates empty groups
                        high_mask_test = gene_values >= test_threshold
                        low_mask_test = gene_values < test_threshold
                        
                        if high_mask_test.sum() < 5 or low_mask_test.sum() < 5:
                            continue
                        
                        # Perform log-rank test
                        results = logrank_test(
                            durations[high_mask_test], 
                            durations[low_mask_test],
                            event_observed_A=events[high_mask_test], 
                            event_observed_B=events[low_mask_test]
                        )
                        
                        # Update best threshold if p-value is lower
                        if results.p_value < best_pvalue:
                            best_pvalue = results.p_value
                            best_threshold = test_threshold
                    
                    threshold = best_threshold
                    print(f"Optimal threshold for {gene}: {threshold:.4f} (p-value: {best_pvalue:.4e})")
                
                high_mask = gene_values >= threshold
                low_mask = gene_values < threshold
                
                groups = [
                    ('High', high_mask, group_colors[0]),
                    ('Low', low_mask, group_colors[1])
                ]
                
            else:
                raise ValueError("split_method must be one of 'median', 'quartile', 'tertile', or 'optimal'")
            
            # Store p-values for log-rank tests
            pvalues = {}
            
            # Fit and plot Kaplan-Meier curves for each group
            kmf_list = []
            
            # Perform multivariate log-rank test if more than 2 groups
            if len(groups) > 2 and all(mask.sum() >= 2 for _, mask, _ in groups):
                # Prepare data for multivariate test
                multi_durations = durations.copy()
                multi_events = events.copy()
                
                # Create a group label column (using integers for group labels)
                multi_groups = np.zeros(len(durations))
                for group_idx, (_, mask, _) in enumerate(groups):
                    multi_groups[mask] = group_idx + 1
                
                # Perform multivariate log-rank test
                multi_results = multivariate_logrank_test(
                    multi_durations,
                    multi_groups,
                    multi_events
                )
                
                # Store the overall p-value
                pvalues['Overall'] = multi_results.p_value
            
            for idx, (label, mask, color) in enumerate(groups):
                if mask.sum() < 2:  # Skip if too few samples
                    print(f"Warning: Group '{label}' for gene {gene} has fewer than 2 samples. Skipping.")
                    continue
                
                # Get subset of data for this group
                group_durations = durations[mask]
                group_events = events[mask]
                
                # Create Kaplan-Meier fitter and fit data
                kmf = KaplanMeierFitter()
                kmf.fit(
                    group_durations, 
                    group_events, 
                    label=f"{label} (n={mask.sum()})"
                )
                
                # Plot the KM curve
                kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
                kmf_list.append((label, kmf, mask))
            
            # Perform log-rank test for each pair of groups
            if len(kmf_list) >= 2:
                for i in range(len(kmf_list)):
                    for j in range(i+1, len(kmf_list)):
                        label_i, _, mask_i = kmf_list[i]
                        label_j, _, mask_j = kmf_list[j]
                        
                        # Calculate log-rank test
                        results = logrank_test(
                            durations[mask_i], 
                            durations[mask_j],
                            event_observed_A=events[mask_i], 
                            event_observed_B=events[mask_j]
                        )
                        
                        # Store p-value
                        pvalues[f"{label_i} vs {label_j}"] = results.p_value
            
            # Add labels and title for the subplot
            ax.set_xlabel(f"Time ({time_unit})", fontsize=12)
            ax.set_ylabel("Survival Probability", fontsize=12)
            
            # Format p-values for title
            if pvalues:
                # Format main p-value (first pair or median split) for the title
                main_pair = list(pvalues.keys())[0]
                p_value = pvalues[main_pair]
                p_formatted = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
                
                title = f"{gene}\nGrouped by {data_label}, {p_formatted}"
            else:
                title = f"{gene}\nGrouped by {data_label}"
                
            ax.set_title(title, fontsize=14)
            
            # Add a grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set y-axis limits from 0 to 1
            ax.set_ylim(0, 1.05)
            
            # Add p-values as text in the plot
            if pvalues:
                p_text = "\n".join([f"{pair}: {p:.4f}" if p >= 0.0001 else f"{pair}: p < 0.0001" 
                                  for pair, p in pvalues.items()])
                ax.text(0.05, 0.05, p_text, transform=ax.transAxes,
                      fontsize=10, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add at-risk table if requested
            if at_risk_table and len(kmf_list) > 0:
                # Determine time points for at-risk table
                max_time = durations.max()
                
                # Define time points based on the range of the data
                if max_time <= 100:
                    # For short follow-up periods
                    time_points = np.linspace(0, max_time, 5, dtype=int)
                elif max_time <= 365 * 2:  # If up to 2 years
                    # Use months as the interval
                    n_intervals = 8
                    time_points = np.linspace(0, max_time, n_intervals, dtype=int)
                else:
                    # For longer follow-up periods, use years equivalent
                    years = int(max_time / 365)
                    time_points = np.array([i * 365 for i in range(years + 1)])
                
                # Calculate number of patients at risk at each time point
                at_risk_data = []
                
                for label, kmf, mask in kmf_list:
                    # Get KM survival function values at each time point
                    risk_counts = []
                    
                    for t in time_points:
                        # Calculate number at risk at time t
                        count = sum((durations[mask] >= t))
                        risk_counts.append(count)
                    
                    at_risk_data.append((label, risk_counts))
                
                # Create the at-risk table below the plot
                # Create a new axes for the at-risk table below the KM plot
                table_height = 0.15  # Height of the table as a fraction of the plot height
                
                # Get the position of the KM plot
                bbox = ax.get_position()
                
                # Create a new axes for the at-risk table
                table_ax = fig.add_axes([
                    bbox.x0,
                    bbox.y0 - table_height,
                    bbox.width,
                    table_height
                ])
                
                # Hide the axes elements
                table_ax.axis('off')
                
                # Create a formatted version of the time points
                time_labels = [f"{int(t)}" for t in time_points]
                
                # Prepare table data
                table_data = []
                table_data.append(["At risk"] + time_labels)
                
                for label, counts in at_risk_data:
                    table_data.append([label] + [str(count) for count in counts])
                
                # Create the table
                table = table_ax.table(
                    cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2] + [0.8/len(time_points)] * len(time_points)
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                
                # Remember the table axis for adjustments
                all_axes.append(table_ax)
        
        # Add a main title for the figure
        fig.suptitle(f"{title_prefix} Cancer: Kaplan-Meier Survival Analysis by {data_type.upper()} Values",
                    fontsize=18, y=0.98)
        
        # Remove any unused subplots
        if layout == 'grid':
            for j in range(n_genes, len(axes)):
                fig.delaxes(axes[j])
        
        # Adjust layout considering at-risk tables
        fig.tight_layout()
        if layout == 'grid':
            if at_risk_table:
                plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.4)  # More space at bottom for at-risk tables
            else:
                plt.subplots_adjust(top=0.9, bottom=0.1)  # Standard adjustment
        else:
            if at_risk_table:
                plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.6)  # More space between subplots for at-risk tables
            else:
                plt.subplots_adjust(top=0.95, hspace=0.4)  # Standard adjustment
        
        print("Displaying plot...")
        plt.show()
        
        return fig

if __name__ == "__main__":
    data_dir = "data"
    analyzer = OmicsAnalyzer(data_dir)
    
    analyzer.plot_kaplan_meier_by_gene(
        cancer_type='colorec', 
        data_type='rnaseq', 
        n_top_genes=5,
        split_method='median',
        at_risk_table=False
    )



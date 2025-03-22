import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import random
import time
import concurrent.futures
import math
import networkx as nx

import requests
from core_biogrid import config as cfg

class OmicsAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.clorec_dir = os.path.join(data_dir, "colorec/omics_data")
        self.clorec_paths = glob.glob(os.path.join(self.clorec_dir, "*.csv"))
        self.panc_dir = os.path.join(data_dir, "panc/omics_data")
        self.panc_paths = glob.glob(os.path.join(self.panc_dir, "*.csv"))
        self.colorec_omics = {
            "clinical": pd.read_csv(self.clorec_paths[0]),
            "methylation": pd.read_csv(self.clorec_paths[1]),
            "miRNA": pd.read_csv(self.clorec_paths[2]),
            "rnaseq": pd.read_csv(self.clorec_paths[3]),
            "scnv": pd.read_csv(self.clorec_paths[4])
            }
        self.panc_omics = {
            "clinical": pd.read_csv(self.panc_paths[0]),
            "methylation": pd.read_csv(self.panc_paths[1]),
            "miRNA": pd.read_csv(self.panc_paths[2]),
            "rnaseq": pd.read_csv(self.panc_paths[3]),
            "scnv": pd.read_csv(self.panc_paths[4])
            }
        self.harmonized_colorec = self.harmonize_omics("colorectal")
        self.harmonized_panc = self.harmonize_omics("pancreatic")

        self.biogrid_request_url = cfg.BASE_URL + "/interactions"
        
        # Setup for colorectal cancer
        gene_list_methylation_colorec = list(set(self.colorec_omics["methylation"].columns[1:]))
        gene_list_rnaseq_colorec = list(set(self.colorec_omics["rnaseq"].columns[1:]))
        self.gene_list_colorec = list(set(gene_list_methylation_colorec).union(gene_list_rnaseq_colorec))
        
        # Setup for pancreatic cancer
        gene_list_methylation_panc = list(set(self.panc_omics["methylation"].columns[1:]))
        gene_list_rnaseq_panc = list(set(self.panc_omics["rnaseq"].columns[1:]))
        self.gene_list_panc = list(set(gene_list_methylation_panc).union(gene_list_rnaseq_panc))

        # Create interaction data directories if they don't exist
        self.colorec_interaction_dir = os.path.join(data_dir, "colorec/interaction_data")
        self.panc_interaction_dir = os.path.join(data_dir, "panc/interaction_data")
        os.makedirs(self.colorec_interaction_dir, exist_ok=True)
        os.makedirs(self.panc_interaction_dir, exist_ok=True)

        self.biogrid_params_colorec = {
            "accessKey": cfg.ACCESS_KEY,
            "geneList": "|".join(self.gene_list_colorec),
            "format": "tab2",
            "taxId": 9606, # human
            "includeInteractors": "false", # only get interactions between the genes in the gene_list,
            "includeHeader": "true",
            "interSpeciesExcluded": "true", # exclude interactions between different species
            "searchSynonyms": "true",
            "selfInteractionsExcluded": "true" # exclude interactions between the same gene
        }
        
        self.biogrid_params_panc = {
            "accessKey": cfg.ACCESS_KEY,
            "geneList": "|".join(self.gene_list_panc),
            "format": "tab2",
            "taxId": 9606, # human
            "includeInteractors": "false", # only get interactions between the genes in the gene_list,
            "includeHeader": "true",
            "interSpeciesExcluded": "true", # exclude interactions between different species
            "searchSynonyms": "true",
            "selfInteractionsExcluded": "true" # exclude interactions between the same gene
        }
    
    def get_biogrid_interactions(self, batch_size=100, max_workers=5, delay=0.2):
        """
        Get BioGRID interactions for genes in batches with parallel processing.
        
        Args:
            batch_size: Number of genes per batch
            max_workers: Maximum number of parallel requests
            delay: Delay between batches in seconds
        """
        # Process both cancer types
        for cancer_type in ["colorectal", "pancreatic"]:
            print(f"\nAcquiring interactions for {cancer_type} cancer from BioGRID...")
            
            # Set appropriate variables based on cancer type
            if cancer_type == "colorectal":
                gene_list = self.gene_list_colorec
                biogrid_params = self.biogrid_params_colorec
                interaction_dir = self.colorec_interaction_dir
            else:  # pancreatic
                gene_list = self.gene_list_panc
                biogrid_params = self.biogrid_params_panc
                interaction_dir = self.panc_interaction_dir
            
            # First test with small random samples to check API response
            print(f"Testing API with small random samples for {cancer_type} cancer...")
            success_count = 0
            for i in range(10):
                # Select 5 random genes from the gene list
                sample_genes = random.sample(gene_list, 5)
                test_params = biogrid_params.copy()
                test_params["geneList"] = "|".join(sample_genes)
                
                print(f"Test {i+1}/10: Testing with genes: {', '.join(sample_genes)}")
                test_response = requests.get(self.biogrid_request_url, params=test_params)
                
                if test_response.status_code == 200:
                    success_count += 1
                    print(f"✓ Test {i+1} successful")
                else:
                    print(f"✗ Test {i+1} failed: {test_response.status_code}")
                    print(test_response.text)
                
                # Wait 200ms between requests
                time.sleep(delay)
            
            print(f"Small sample tests: {success_count}/10 successful")
            
            if success_count < 8:
                print(f"Too many test failures for {cancer_type} cancer. Please check your API connection and try again.")
                continue  # Skip to next cancer type
                
            # Process genes in batches
            total_genes = len(gene_list)
            num_batches = math.ceil(total_genes / batch_size)
            print(f"Processing {total_genes} genes in {num_batches} batches (batch size: {batch_size})")
            
            # Function to process a single batch
            def process_batch(batch_idx):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_genes)
                batch_genes = gene_list[start_idx:end_idx]
                
                batch_params = biogrid_params.copy()
                batch_params["geneList"] = "|".join(batch_genes)
                
                print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_genes)} genes)")
                try:
                    response = requests.get(self.biogrid_request_url, params=batch_params)
                    
                    if response.status_code == 200:
                        batch_filename = os.path.join(interaction_dir, f"biogrid_batch_{batch_idx+1}.tab2")
                        with open(batch_filename, "w") as f:
                            f.write(response.text)
                        print(f"✓ Batch {batch_idx+1} successful - saved to {batch_filename}")
                        return (True, batch_filename, None)
                    else:
                        print(f"✗ Batch {batch_idx+1} failed: {response.status_code}")
                        return (False, None, f"HTTP {response.status_code}: {response.text[:100]}...")
                except Exception as e:
                    print(f"✗ Batch {batch_idx+1} error: {str(e)}")
                    return (False, None, str(e))
            
            # Process batches with parallel execution
            successful_files = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches with a small delay between submissions
                futures = []
                for batch_idx in range(num_batches):
                    futures.append(executor.submit(process_batch, batch_idx))
                    time.sleep(delay)  # Add delay between batch submissions
                    
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    success, filename, error = future.result()
                    if success:
                        successful_files.append(filename)
            
            # Combine all successful batch files and convert from tab2 to CSV
            if successful_files:
                print(f"Successfully processed {len(successful_files)} of {num_batches} batches")
                
                # Create pandas DataFrame to hold all data
                all_data = []
                header = None
                
                for filename in successful_files:
                    try:
                        # Read tab2 file (tab-delimited)
                        with open(filename, "r") as f:
                            lines = f.readlines()
                            if lines:
                                # Process header (first line) which starts with #
                                if not header and lines[0].startswith('#'):
                                    header = lines[0].strip('#').strip().split('\t')
                                    
                                # Process data lines (skip header if present)
                                start_idx = 1 if lines[0].startswith('#') else 0
                                for line in lines[start_idx:]:
                                    if line.strip():  # Skip empty lines
                                        all_data.append(line.strip().split('\t'))
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
                
                if header and all_data:
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(all_data, columns=header)
                    
                    # Write to CSV file
                    csv_file = os.path.join(interaction_dir, f"{cancer_type}_biogrid_interactions.csv")
                    df.to_csv(csv_file, index=False)
                    
                    print(f"{cancer_type.capitalize()} BioGRID interactions converted to CSV and saved to {csv_file}")
                    print(f"Total interactions: {len(df)}")
                else:
                    print(f"No data or header found in the batch files for {cancer_type} cancer.")
                
                # Optionally clean up batch files
                for filename in successful_files:
                    os.remove(filename)
                print("Temporary batch files cleaned up")
            else:
                print(f"No successful batches for {cancer_type} cancer. Failed to retrieve BioGRID interactions.")

    def initial_exploration(self, cancer_type):
        print(f"Initial exploration of {cancer_type} omics data:")
        # get the methylation data
        if cancer_type == "colorectal":
            methylation = self.harmonized_colorec["methylation"]
        elif cancer_type == "pancreatic":
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
        if cancer_type == "colorectal":
            miRNA = self.harmonized_colorec["miRNA"]
        elif cancer_type == "pancreatic":
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
        if cancer_type == "colorectal":
            rnaseq = self.harmonized_colorec["rnaseq"]
        elif cancer_type == "pancreatic":
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
        if cancer_type == "colorectal":
            scnv = self.harmonized_colorec["scnv"]
        elif cancer_type == "pancreatic":
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
        if cancer_type == "colorectal":
            clinical = self.colorec_omics["clinical"]
            methylation = self.colorec_omics["methylation"]
            miRNA = self.colorec_omics["miRNA"]
            rnaseq = self.colorec_omics["rnaseq"]
            scnv = self.colorec_omics["scnv"]
        elif cancer_type == "pancreatic":
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
            interaction_file = os.path.join(self.colorec_interaction_dir, "colorectal_biogrid_interactions.csv")
            title = "Colorectal Cancer Gene Interaction Network"
        elif cancer_type == 'panc':
            interaction_file = os.path.join(self.panc_interaction_dir, "pancreatic_biogrid_interactions.csv")
            title = "Pancreatic Cancer Gene Interaction Network"
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
        if 'Systematic Name Interactor A' in interactions_df.columns and 'Systematic Name Interactor B' in interactions_df.columns:
            # Create a new graph
            G = nx.Graph()
            
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
                pos = nx.spring_layout(G, seed=42)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G, seed=42)
            else:
                pos = nx.spring_layout(G, seed=42)  # default to spring layout
            
            # Draw the graph
            nx.draw_networkx(
                G, 
                pos=pos,
                node_size=50,
                node_color='skyblue',
                edge_color='gray',
                alpha=0.7,
                with_labels=False  # Labels can make the graph cluttered with many nodes
            )
            
            # Add title and other visual elements
            plt.title(title)
            plt.axis('off')  # Hide axis
            
            # Show the plot
            plt.tight_layout()
            plt.show()
            
            return G
        else:
            print("Required columns 'Systematic Name Interactor A' and 'Systematic Name Interactor B' not found in the interaction data.")
            print("Available columns:", interactions_df.columns.tolist())
            return None

if __name__ == "__main__":
    data_dir = "data"
    analyzer = OmicsAnalyzer(data_dir)
    
    #analyzer.get_biogrid_interactions()
    
    analyzer.visualize_interaction_network(cancer_type='colorec')
    
    analyzer.visualize_interaction_network(cancer_type='panc')


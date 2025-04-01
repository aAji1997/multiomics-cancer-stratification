import os
import csv
import glob
import pandas as pd
import requests
from core_biogrid import config as cfg
import random
import time
import concurrent.futures
import math
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
# Try to import tqdm for progress bars, use a simple fallback if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import the InteractionProcessor class
from preprocessing.interaction_processor import InteractionProcessor

class DataExtractor:
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
        self.harmonized_colorec = self.harmonize_omics("colorec")
        self.harmonized_panc = self.harmonize_omics("panc")

        self.biogrid_request_url = cfg.BASE_URL + "/interactions"
        
        # Setup for colorec cancer
        gene_list_methylation_colorec = list(set(self.colorec_omics["methylation"].columns[1:]))
        gene_list_rnaseq_colorec = list(set(self.colorec_omics["rnaseq"].columns[1:]))
        gene_list_scnv_colorec = list(set(self.colorec_omics["scnv"].columns[1:]))
        self.gene_list_colorec = list(set(gene_list_methylation_colorec).union(gene_list_rnaseq_colorec).union(gene_list_scnv_colorec))
        
        # Setup for panc cancer
        gene_list_methylation_panc = list(set(self.panc_omics["methylation"].columns[1:]))
        gene_list_rnaseq_panc = list(set(self.panc_omics["rnaseq"].columns[1:]))
        gene_list_scnv_panc = list(set(self.panc_omics["scnv"].columns[1:]))
        self.gene_list_panc = list(set(gene_list_methylation_panc).union(gene_list_rnaseq_panc).union(gene_list_scnv_panc))

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
            "includeInteractors": "true", # Get interactions involving at least one gene in the list
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
            "includeInteractors": "true", # Get interactions involving at least one gene in the list
            "includeHeader": "true",
            "interSpeciesExcluded": "true", # exclude interactions between different species
            "searchSynonyms": "true",
            "selfInteractionsExcluded": "true" # exclude interactions between the same gene
        }
        # Try to load the interaction files, if they don't exist, fetch them from BioGRID
        colorec_file_path = os.path.join(self.colorec_interaction_dir, "colorec_biogrid_interactions.csv")
        panc_file_path = os.path.join(self.panc_interaction_dir, "panc_biogrid_interactions.csv")
        
        try:
            self.biogrid_interactions_colorec = pd.read_csv(colorec_file_path)
            self.biogrid_interactions_panc = pd.read_csv(panc_file_path)
        except FileNotFoundError:
            print("BioGRID interaction files not found. Fetching from BioGRID API...")
            self.get_biogrid_interactions()
            # Try loading again after fetching
            self.biogrid_interactions_colorec = pd.read_csv(colorec_file_path)
            self.biogrid_interactions_panc = pd.read_csv(panc_file_path)
    
    def check_gene_overlap(self):
        """
        Checks the overlap of genes present in methylation, rnaseq, and scnv data
        for both colorec and panc cancer types.

        Returns:
            dict: Dictionary containing overlap statistics for both cancer types.
        """
        results = {}
        print("\n--- Checking Gene Overlap Across Omics Data ---")

        for cancer_type in ["colorec", "panc"]:
            print(f"\nAnalyzing {cancer_type.capitalize()} Cancer Data:")

            # Access the correct omics data dictionary
            omics_data = self.colorec_omics if cancer_type == "colorec" else self.panc_omics

            # Extract gene sets (column names excluding 'patient_id')
            try:
                methylation_genes = set(omics_data["methylation"].columns[1:])
                rnaseq_genes = set(omics_data["rnaseq"].columns[1:])
                scnv_genes = set(omics_data["scnv"].columns[1:])
            except KeyError as e:
                print(f"Error: Missing omics data file for {cancer_type}: {e}")
                results[cancer_type] = {"error": f"Missing data: {e}"}
                continue # Skip to the next cancer type

            # Calculate intersections and unions
            common_genes = methylation_genes.intersection(rnaseq_genes).intersection(scnv_genes)
            all_genes = methylation_genes.union(rnaseq_genes).union(scnv_genes)

            # Calculate pairwise intersections
            meth_rna_common = methylation_genes.intersection(rnaseq_genes)
            meth_scnv_common = methylation_genes.intersection(scnv_genes)
            rna_scnv_common = rnaseq_genes.intersection(scnv_genes)

            # Calculate overlap percentage
            total_unique_genes = len(all_genes)
            common_to_all = len(common_genes)
            overlap_percent = (common_to_all / total_unique_genes) * 100 if total_unique_genes > 0 else 0

            # Store results
            results[cancer_type] = {
                "methylation_gene_count": len(methylation_genes),
                "rnaseq_gene_count": len(rnaseq_genes),
                "scnv_gene_count": len(scnv_genes),
                "total_unique_genes": total_unique_genes,
                "common_genes_count": common_to_all,
                "overlap_percentage": overlap_percent,
                "common_genes_list": sorted(list(common_genes)),
                "pairwise_overlap": {
                    "methylation_rnaseq": len(meth_rna_common),
                    "methylation_scnv": len(meth_scnv_common),
                    "rnaseq_scnv": len(rna_scnv_common)
                }
            }

            # Print summary
            print(f"  Methylation genes: {len(methylation_genes)}")
            print(f"  RNASeq genes: {len(rnaseq_genes)}")
            print(f"  SCNV genes: {len(scnv_genes)}")
            print(f"  Total unique genes across these 3 datasets: {total_unique_genes}")
            print(f"  Genes common to all 3 datasets: {common_to_all}")
            print(f"  Overlap percentage (common / total unique): {overlap_percent:.2f}%")
            print(f"  Pairwise Overlap:")
            print(f"    Methylation & RNASeq: {len(meth_rna_common)}")
            print(f"    Methylation & SCNV: {len(meth_scnv_common)}")
            print(f"    RNASeq & SCNV: {len(rna_scnv_common)}")

        print("--- Gene Overlap Check Complete ---")
        return results

    def check_gene_coverage_in_interactions(self):
        """
        Check whether genes in gene_list_colorec and gene_list_panc appear in the respective
        BioGRID interaction dataframes, either as systematic names, official symbols, or synonyms.
        
        Returns:
            dict: Dictionary containing coverage statistics and lists of missing genes for both cancer types
        """
        results = {}
        
        # Process both cancer types
        for cancer_type in ["colorec", "panc"]:
            print(f"\nChecking gene coverage in {cancer_type} cancer interactions...")
            
            # Set appropriate variables based on cancer type
            if cancer_type == "colorec":
                gene_list = self.gene_list_colorec
                interactions_df = self.biogrid_interactions_colorec
                key_prefix = "colorec"
            else:  # panc
                gene_list = self.gene_list_panc
                interactions_df = self.biogrid_interactions_panc
                key_prefix = "panc"
                
            # Get all systematic names from the interaction data
            systematic_names = set()
            if 'Systematic Name Interactor A' in interactions_df.columns:
                systematic_names.update(interactions_df['Systematic Name Interactor A'].dropna().unique())
            if 'Systematic Name Interactor B' in interactions_df.columns:
                systematic_names.update(interactions_df['Systematic Name Interactor B'].dropna().unique())
            
            # Remove empty and placeholder values
            systematic_names = {name for name in systematic_names if name and name != '-'}
            
            # Get all official symbols from the interaction data
            official_symbols = set()
            if 'Official Symbol Interactor A' in interactions_df.columns:
                official_symbols.update(interactions_df['Official Symbol Interactor A'].dropna().unique())
            if 'Official Symbol Interactor B' in interactions_df.columns:
                official_symbols.update(interactions_df['Official Symbol Interactor B'].dropna().unique())
                
            # Remove empty and placeholder values
            official_symbols = {symbol for symbol in official_symbols if symbol and symbol != '-'}
            
            # Get all synonym names from the interaction data
            synonym_lists = []
            if 'Synonyms Interactor A' in interactions_df.columns:
                synonym_lists.extend(interactions_df['Synonyms Interactor A'].dropna().unique())
            if 'Synonyms Interactor B' in interactions_df.columns:
                synonym_lists.extend(interactions_df['Synonyms Interactor B'].dropna().unique())
            
            # Process synonym lists (they're pipe-separated)
            all_synonyms = set()
            for syn_list in synonym_lists:
                if pd.notna(syn_list) and syn_list and syn_list != '-':
                    synonyms = [s.strip() for s in syn_list.split('|')]
                    all_synonyms.update(synonyms)
            
            # Check which genes from the gene list appear in the interaction data
            found_in_systematic = set()
            found_in_official = set()
            found_in_synonyms = set()
            missing_genes = set()
            
            for gene in gene_list:
                if gene in systematic_names:
                    found_in_systematic.add(gene)
                elif gene in official_symbols:
                    found_in_official.add(gene)
                elif gene in all_synonyms:
                    found_in_synonyms.add(gene)
                else:
                    missing_genes.add(gene)
            
            # Calculate coverage statistics
            total_genes = len(gene_list)
            found_genes = len(found_in_systematic) + len(found_in_official) + len(found_in_synonyms)
            coverage_percent = (found_genes / total_genes) * 100 if total_genes > 0 else 0
            
            # Store results
            results[f"{key_prefix}_total_genes"] = total_genes
            results[f"{key_prefix}_found_in_systematic"] = len(found_in_systematic)
            results[f"{key_prefix}_found_in_official"] = len(found_in_official)
            results[f"{key_prefix}_found_in_synonyms"] = len(found_in_synonyms)
            results[f"{key_prefix}_missing_genes"] = len(missing_genes)
            results[f"{key_prefix}_coverage_percent"] = coverage_percent
            results[f"{key_prefix}_found_in_systematic_list"] = list(found_in_systematic)
            results[f"{key_prefix}_found_in_official_list"] = list(found_in_official)
            results[f"{key_prefix}_found_in_synonyms_list"] = list(found_in_synonyms)
            results[f"{key_prefix}_missing_genes_list"] = list(missing_genes)
            
            # Print summary
            print(f"Total genes in {cancer_type} gene list: {total_genes}")
            print(f"Genes found in systematic names: {len(found_in_systematic)} ({len(found_in_systematic)/total_genes*100:.2f}%)")
            print(f"Genes found in official symbols: {len(found_in_official)} ({len(found_in_official)/total_genes*100:.2f}%)")
            print(f"Genes found in synonyms: {len(found_in_synonyms)} ({len(found_in_synonyms)/total_genes*100:.2f}%)")
            print(f"Total coverage: {coverage_percent:.2f}%")
            print(f"Missing genes: {len(missing_genes)} ({len(missing_genes)/total_genes*100:.2f}%)")
            
            # Print some examples of missing genes if any
            if missing_genes:
                example_count = min(5, len(missing_genes))
                print(f"Examples of missing genes: {list(missing_genes)[:example_count]}")
        
        return results
    
    def get_biogrid_interactions(self, batch_size=100, max_workers=5, delay=0.2):
        """
        Get BioGRID interactions for genes in batches with parallel processing.
        
        Args:
            batch_size: Number of genes per batch
            max_workers: Maximum number of parallel requests
            delay: Delay between batches in seconds
        """
        # Process both cancer types
        for cancer_type in ["colorec", "panc"]:
            print(f"\nAcquiring interactions for {cancer_type} cancer from BioGRID...")
            
            # Set appropriate variables based on cancer type
            if cancer_type == "colorec":
                gene_list = self.gene_list_colorec
                biogrid_params = self.biogrid_params_colorec
                interaction_dir = self.colorec_interaction_dir
            else:  # panc
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

    def convert_to_csv(self):
        """
        Recursively converts all continuous data matrix (cct) text files within the folder_path 
        and its subfolders to CSV format and saves them in the same directory as the source files.
        
        The cct file format has variable whitespace separation between attributes with attributes as rows 
        and samples as columns. The first column contains attribute names, and the first row contains sample IDs.
        The data is transposed during conversion, so rows become columns and vice versa.
        
        Returns:
            list: Paths to the generated CSV files
        """
        
        converted_files = []
        
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    
                    # Get the directory and filename without extension
                    directory = os.path.dirname(file_path)
                    filename_without_ext = os.path.splitext(filename)[0]
                    
                    # Create output path with csv extension
                    output_path = os.path.join(directory, f"{filename_without_ext}.csv")
                    
                    try:
                        # Read the entire text file into memory
                        with open(file_path, 'r') as input_file:
                            # Read all lines and split by any whitespace (variable spacing)
                            rows = [line.strip().split() for line in input_file]
                        
                        # Transpose the data (swap rows and columns)
                        transposed_data = list(zip(*rows))
                        
                        #Change 'attrib_name' to 'patient_id'
                        if transposed_data and transposed_data[0][0] == 'attrib_name':
                            transposed_data[0] = ('patient_id',) + transposed_data[0][1:]
                        
                        # Write the transposed data to CSV
                        with open(output_path, 'w', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            for row in transposed_data:
                                csv_writer.writerow(row)
                        
                        print(f"Successfully converted and transposed {file_path} to {output_path}")
                        converted_files.append(output_path)
                            
                    except Exception as e:
                        print(f"Error converting file {file_path}: {e}")
        
        return converted_files
    
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
    
    def preprocess_omics(self):
        """
        Scale all omics data for both cancer types.
        Returns scaled DataFrames with preserved patient IDs.
        """
        processed_data = {}
        
        for cancer_type in ["colorec", "panc"]:
            cancer_data = self.harmonized_colorec if cancer_type == "colorec" else self.harmonized_panc
            processed_cancer = {}
            
            # Process each omics data type
            for omics_type, df in cancer_data.items():
                if omics_type == "clinical":
                    # Preserve clinical data for later survival analysis
                    processed_cancer[omics_type] = df.copy()
                    continue
                    
                # Save patient IDs
                patient_ids = df["patient_id"].values
                
                # Get features for scaling
                features = df.drop("patient_id", axis=1)
                
                # Scale the features
                scaler_std = StandardScaler()
                scaler_minmax = MinMaxScaler()
                scaled_features_std = scaler_std.fit_transform(features)
                scaled_features_minmax = scaler_minmax.fit_transform(scaled_features_std)
                
                scaled_df_minmax = pd.DataFrame(scaled_features_minmax, columns=features.columns)
                scaled_df_minmax.insert(0, "patient_id", patient_ids)
                
                processed_cancer[omics_type] = scaled_df_minmax
                
            processed_data[cancer_type] = processed_cancer
        
        return processed_data
    
    def process_interaction_data(self, cancer_type="both", identifier_type="Official Symbol"):
        """
        Process interaction data for GCN using the InteractionProcessor.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            identifier_type (str): Type of gene identifier to use
            
        Returns:
            dict: Dictionary with processing results or None if processing failed
        """
        results = {}
        
        if cancer_type in ["colorec", "both"]:
            print("\nProcessing colorec cancer interaction data:")
            colorec_file = os.path.join(self.colorec_interaction_dir, "colorec_biogrid_interactions.csv")
            processor = InteractionProcessor(
                interaction_dir=self.colorec_interaction_dir,
                gene_list=self.gene_list_colorec,
                identifier_type=identifier_type
            )
            
            # Check if already processed
            processed_dir = os.path.join(self.colorec_interaction_dir, "processed")
            processed_files = glob.glob(os.path.join(processed_dir, "colorec_*.npz"))
            
            if processed_files:
                print("Found existing processed data. Loading...")
                results["colorec"] = processor.load_processed_data("colorec")
            else:
                # Process the data
                results["colorec"] = processor.process_interactions(
                    colorec_file, 
                    "colorec",
                    add_self_loops=True,
                    normalize=True
                )
        
        if cancer_type in ["panc", "both"]:
            print("\nProcessing panc cancer interaction data:")
            panc_file = os.path.join(self.panc_interaction_dir, "panc_biogrid_interactions.csv")
            processor = InteractionProcessor(
                interaction_dir=self.panc_interaction_dir,
                gene_list=self.gene_list_panc,
                identifier_type=identifier_type
            )
            
            # Check if already processed
            processed_dir = os.path.join(self.panc_interaction_dir, "processed")
            processed_files = glob.glob(os.path.join(processed_dir, "panc_*.npz"))
            
            if processed_files:
                print("Found existing processed data. Loading...")
                results["panc"] = processor.load_processed_data("panc")
            else:
                # Process the data
                results["panc"] = processor.process_interactions(
                    panc_file, 
                    "panc",
                    add_self_loops=True,
                    normalize=True
                )
        
        return results
    
    def harmonize_genes_across_omics(self, cancer_type="both"):
        """
        Create consistent gene sets across all omics data types.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            
        Returns:
            dict: Dictionary containing harmonized omics data for each cancer type
        """
        results = {}
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nHarmonizing genes across omics types for {cancer} cancer...")
            
            # Get the appropriate omics data
            if cancer == "colorec":
                omics_data = self.harmonized_colorec
            else:  # panc
                omics_data = self.harmonized_panc
            
            # Get all unique genes across omics types
            all_genes = set()
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    all_genes.update(set(omics_data[omics_type].columns[1:]))
            
            print(f"Found {len(all_genes)} unique genes across all omics types")
            
            # Create harmonized dataframes with consistent gene columns
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    # Create a proper copy to avoid SettingWithCopyWarning
                    df = omics_data[omics_type].copy()
                    current_genes = set(df.columns[1:])
                    missing_genes = all_genes - current_genes
                    
                    print(f"Adding {len(missing_genes)} missing genes to {omics_type} data")
                    
                    if missing_genes:
                        # Create a DataFrame with zeros for all missing genes
                        # This is more efficient than adding columns one by one
                        missing_df = pd.DataFrame(0.0, 
                                              index=df.index, 
                                              columns=list(missing_genes))
                        
                        # Concatenate with original DataFrame along columns
                        df = pd.concat([df, missing_df], axis=1)
                    
                    # Update the dataframe in the omics dictionary
                    omics_data[omics_type] = df
            
            results[cancer] = omics_data
        
        # Update the harmonized data in the class
        if "colorec" in results:
            self.harmonized_colorec = results["colorec"]
        if "panc" in results:
            self.harmonized_panc = results["panc"]
            
        return results
    
    def handle_missing_values(self, strategy="zero", cancer_type="both"):
        """
        Handle missing values in harmonized omics data.
        
        Args:
            strategy (str): Strategy for handling missing values, one of:
                           "zero" - fill with zeros
                           "mean" - fill with column mean
                           "median" - fill with column median
            cancer_type (str): "colorec", "panc", or "both"
            
        Returns:
            dict: Dictionary containing omics data with missing values handled
        """
        results = {}
        strategies = {
            "zero": 0.0,
            "mean": lambda col: col.mean(),
            "median": lambda col: col.median()
        }
        
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {list(strategies.keys())}")
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nHandling missing values in {cancer} cancer data using {strategy} strategy...")
            
            # Get the appropriate omics data
            if cancer == "colorec":
                omics_data = self.harmonized_colorec
            else:  # panc
                omics_data = self.harmonized_panc
            
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    df = omics_data[omics_type]
                    
                    # Count missing values
                    missing_count = df.isna().sum().sum()
                    if missing_count > 0:
                        print(f"Found {missing_count} missing values in {omics_type} data")
                        
                        # Fill missing values based on strategy
                        if strategy == "zero":
                            df = df.fillna(0.0)
                        elif strategy in ["mean", "median"]:
                            fill_func = strategies[strategy]
                            # Apply the function column-wise, excluding patient_id
                            for col in df.columns[1:]:
                                df[col] = df[col].fillna(fill_func(df[col]))
                                
                        # Update the dataframe
                        omics_data[omics_type] = df
            
            results[cancer] = omics_data
        
        # Update the harmonized data in the class
        if "colorec" in results:
            self.harmonized_colorec = results["colorec"]
        if "panc" in results:
            self.harmonized_panc = results["panc"]
            
        return results
    
    def create_gene_masks(self, cancer_type="both"):
        """
        Create binary masks indicating gene presence in each omics type.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            
        Returns:
            dict: Dictionary containing gene masks for each cancer and omics type
        """
        all_masks = {}
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nCreating gene masks for {cancer} cancer...")
            
            # Get the appropriate omics data
            if cancer == "colorec":
                omics_data = self.harmonized_colorec
            else:  # panc
                omics_data = self.harmonized_panc
            
            # Get all unique genes across all omics types
            all_genes = set()
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    all_genes.update(set(omics_data[omics_type].columns[1:]))
            all_genes = sorted(list(all_genes))
            
            # Create masks for each omics type
            masks = {}
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    df = omics_data[omics_type]
                    original_genes = set(df.columns[1:])
                    
                    # Create binary mask (1 for originally present, 0 for added during harmonization)
                    mask = [1 if gene in original_genes else 0 for gene in all_genes]
                    masks[omics_type] = mask
                    
                    present_count = sum(mask)
                    print(f"{omics_type}: {present_count} genes originally present out of {len(all_genes)}")
            
            all_masks[cancer] = {
                'gene_list': all_genes,
                'masks': masks
            }
        
        # Store masks as a class attribute
        self.gene_masks = all_masks
        return all_masks
    
    def align_genes_with_interactions(self, cancer_type="both", identifier_type="Official Symbol"):
        """
        Ensure genes in omics data align with those in interaction graphs.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            identifier_type (str): Type of gene identifier used in interaction data
            
        Returns:
            dict: Dictionary containing alignment information
        """
        alignment_info = {}
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nAligning genes between omics and interaction data for {cancer} cancer...")
            
            # Process interaction data if not already done
            interaction_data = self.process_interaction_data(cancer, identifier_type)
            if cancer not in interaction_data:
                print(f"No interaction data available for {cancer}")
                continue
                
            # Get the list of genes in the interaction network
            interaction_genes = interaction_data[cancer]['node_list']
            print(f"Interaction network contains {len(interaction_genes)} genes")
            
            # Get the appropriate omics data
            if cancer == "colorec":
                omics_data = self.harmonized_colorec
            else:  # panc
                omics_data = self.harmonized_panc
            
            # Get all unique genes in omics data
            all_omics_genes = set()
            for omics_type in ["methylation", "rnaseq", "scnv"]:
                if omics_type in omics_data:
                    all_omics_genes.update(set(omics_data[omics_type].columns[1:]))
            
            # Find genes in both datasets and genes missing from each
            common_genes = set(interaction_genes) & all_omics_genes
            missing_in_omics = set(interaction_genes) - all_omics_genes
            missing_in_interactions = all_omics_genes - set(interaction_genes)
            
            print(f"Common genes: {len(common_genes)}")
            print(f"Genes in interaction data but not in omics: {len(missing_in_omics)}")
            print(f"Genes in omics but not in interaction data: {len(missing_in_interactions)} ({len(missing_in_interactions)/len(all_omics_genes)*100:.1f}%)")
            
            alignment_info[cancer] = {
                'common_genes': list(common_genes),
                'missing_in_omics': list(missing_in_omics),
                'missing_in_interactions': list(missing_in_interactions),
                'coverage_percentage': len(common_genes) / len(all_omics_genes) * 100
            }
        
        return alignment_info
    
    def impute_missing_interactions(self, cancer_type="both", similarity_metric="cosine", threshold=0.7):
        """
        Impute interactions for genes missing from interaction network based on omics similarity.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            similarity_metric (str): Method to calculate gene similarity, one of:
                                    "correlation" - Pearson correlation
                                    "cosine" - Cosine similarity
            threshold (float): Similarity threshold for creating an edge (0-1)
            
        Returns:
            dict: Dictionary containing interaction data with imputed edges
        """
        import numpy as np
        from scipy.spatial.distance import cosine, pdist, squareform
        from sklearn.preprocessing import StandardScaler
        
        imputed_data = {}
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nImputing missing interactions for {cancer} cancer...")
            
            # Get processed interaction data
            interaction_data = self.process_interaction_data(cancer)
            if cancer not in interaction_data:
                print(f"No interaction data available for {cancer}")
                continue
            
            # Get the appropriate omics data
            if cancer == "colorec":
                omics_data = self.harmonized_colorec
            else:  # panc
                omics_data = self.harmonized_panc
            
            # Combine omics data for better gene similarity calculation
            # Use the average of standardized values across omics types
            combined_omics = {}
            
            for omics_type in ["rnaseq", "methylation", "scnv"]:
                if omics_type in omics_data:
                    df = omics_data[omics_type]
                    patient_ids = df["patient_id"].values
                    
                    # Standardize features
                    std_scaler = StandardScaler()
                    min_max_scaler = MinMaxScaler()
                    features = df.drop("patient_id", axis=1)
                    scaled_features = std_scaler.fit_transform(features)
                    scaled_features = min_max_scaler.fit_transform(scaled_features)
                    
                    # Store as a dict for each gene
                    for i, gene in enumerate(features.columns):
                        if gene not in combined_omics:
                            combined_omics[gene] = []
                        combined_omics[gene].append(scaled_features[:, i])
            
            # Average the values across omics types
            gene_values = {}
            for gene, values_list in combined_omics.items():
                if values_list:  # Check if there's data for this gene
                    gene_values[gene] = np.mean(values_list, axis=0)
            
            # Get interaction network info
            adj_matrix = interaction_data[cancer]['adjacency_matrix']
            # Using 'node_list' instead of 'gene_list'
            interaction_genes = interaction_data[cancer]['node_list']
            
            # Get all genes in omics data
            all_omics_genes = sorted(list(gene_values.keys()))
            print(f"Found {len(all_omics_genes)} genes with omics data")
            print(f"Found {len(interaction_genes)} genes in the interaction network")
            
            # Find genes missing from interaction data
            missing_genes = [g for g in all_omics_genes if g not in interaction_genes]
            print(f"Imputing interactions for {len(missing_genes)} genes missing from interaction network ({len(missing_genes)/len(all_omics_genes)*100:.1f}% of all genes)")
            
            # The gene coverage from BioGRID is approximately 63% as mentioned, so we expect
            # around 37% of genes to need imputation, which explains the high number of "missing" genes
            print("Note: This large number of missing genes is expected due to limited BioGRID coverage (~63%)")
            
            # Create gene-to-index mappings
            all_genes_idx = {gene: idx for idx, gene in enumerate(all_omics_genes)}
            gene_idx_to_name = {idx: gene for gene, idx in all_genes_idx.items()}
            interaction_genes_idx = {gene: idx for idx, gene in enumerate(interaction_genes)}
            
            # Calculate gene-gene similarity matrix using vectorized operations
            print("Calculating gene similarity matrix using vectorized operations...")
            n_genes = len(all_omics_genes)
            
            # Prepare data matrix for vectorized computation
            # First, we need to create a matrix where each row is a gene vector
            data_matrix = np.zeros((n_genes, len(next(iter(gene_values.values())))))
            valid_indices = []
            
            print("Preparing gene data matrix...")
            for gene, idx in all_genes_idx.items():
                if gene in gene_values:
                    data_matrix[idx] = gene_values[gene]
                    valid_indices.append(idx)
            
            valid_indices = np.array(valid_indices)
            valid_data = data_matrix[valid_indices]
            
            # Initialize similarity matrix
            similarity_matrix = np.zeros((n_genes, n_genes))
            
            if similarity_metric == "correlation":
                print("Computing correlation matrix...")
                # Compute the full correlation matrix at once
                if TQDM_AVAILABLE:
                    with tqdm(total=1, desc="Computing correlation matrix", unit="matrix"):
                        # Calculate correlation matrix for valid data
                        valid_corr = np.corrcoef(valid_data)
                        # Convert from [-1, 1] to [0, 1]
                        valid_corr = (valid_corr + 1) / 2
                else:
                    print("Computing correlation matrix... ", end="", flush=True)
                    valid_corr = np.corrcoef(valid_data)
                    valid_corr = (valid_corr + 1) / 2
                    print("Done")
                
                # Copy values to the main similarity matrix
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        similarity_matrix[idx_i, idx_j] = valid_corr[i, j]
                
            elif similarity_metric == "cosine":
                print("Computing cosine similarity matrix using FAISS approximate nearest neighbors...")
                import scipy.sparse as sp
                try:
                    import faiss
                    FAISS_AVAILABLE = True
                except ImportError:
                    print("FAISS not available. Please install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
                    FAISS_AVAILABLE = False
                    
                if not FAISS_AVAILABLE:
                    print("Falling back to standard method. Consider installing FAISS for better performance.")
                    # Calculate pairwise distances
                    distances = pdist(valid_data, metric='cosine')
                    # Convert to square form and then to similarities
                    distance_matrix = squareform(distances)
                    valid_sim = 1 - distance_matrix
                    
                    # Copy values to the main similarity matrix
                    for i, idx_i in enumerate(valid_indices):
                        for j, idx_j in enumerate(valid_indices):
                            similarity_matrix[idx_i, idx_j] = valid_sim[i, j]
                else:
                    print(f"Using FAISS with threshold {threshold} for memory-efficient similarity computation")
                    # Normalize the vectors for cosine similarity
                    normalized_data = valid_data.copy()
                    norms = np.linalg.norm(normalized_data, axis=1, keepdims=True)
                    normalized_data = normalized_data / norms
                    
                    # Convert to float32 for FAISS
                    normalized_data = normalized_data.astype(np.float32)
                    
                    # Build the FAISS index
                    d = normalized_data.shape[1]  # dimension of vectors
                    
                    # For smaller datasets, use exact search
                    if normalized_data.shape[0] < 10000:
                        index = faiss.IndexFlatIP(d)  # inner product = cosine sim for normalized vectors
                    else:
                        # For larger datasets, use approximate search
                        nlist = min(4096, normalized_data.shape[0] // 50)  # number of cells
                        quantizer = faiss.IndexFlatIP(d)
                        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                        # Need to train the index
                        index.train(normalized_data)
                        # Set number of cells to probe at query time
                        index.nprobe = min(256, nlist)
                    
                    # Add vectors to the index
                    index.add(normalized_data)
                    
                    # Search for nearest neighbors
                    # Determine k: number of nearest neighbors to search for
                    k = min(31, normalized_data.shape[0])  # cap at 31 or dataset size
                    
                    # Create sparse matrix for memory-efficient storage
                    row_indices = []
                    col_indices = []
                    similarity_values = []
                    
                    # Search and collect results directly to sparse format
                    for i in range(normalized_data.shape[0]):
                        # Search for k nearest neighbors
                        D, I = index.search(normalized_data[i:i+1], k)
                        
                        # Process results for this point
                        for j, (dist, idx) in enumerate(zip(D[0], I[0])):
                            if idx >= 0:  # Valid index (FAISS may return -1)
                                # dist is actually the inner product, which equals cosine similarity for normalized vectors
                                similarity = float(dist)
                                
                                # Only keep values above threshold
                                if similarity > threshold:
                                    # Map back to original indices
                                    orig_i = valid_indices[i]
                                    orig_j = valid_indices[idx]
                                    
                                    # Add to sparse matrix entries
                                    row_indices.append(orig_i)
                                    col_indices.append(orig_j)
                                    similarity_values.append(similarity)
                    
                    # Create sparse matrix (we'll convert to dense later for compatibility with the rest of the code)
                    sparse_sim = sp.coo_matrix(
                        (similarity_values, (row_indices, col_indices)),
                        shape=(n_genes, n_genes)
                    )
                    
                    # Convert sparse to dense (could be optimized further by modifying the rest of the code)
                    similarity_matrix = sparse_sim.toarray()
                    
                    print(f"Found {len(similarity_values)} similarities above threshold {threshold}")
                    print(f"Sparse matrix density: {len(similarity_values)/(n_genes*n_genes)*100:.4f}%")
            else:
                raise ValueError(f"Invalid similarity metric: {similarity_metric}")
            
            # Handle NaN values in similarity matrix
            nan_count = np.isnan(similarity_matrix).sum()
            if nan_count > 0:
                print(f"Fixing {nan_count} NaN values in similarity matrix...")
                similarity_matrix = np.nan_to_num(similarity_matrix)
            
            # Count high similarity pairs
            high_similarity_count = np.sum(similarity_matrix > threshold)
            
            # Print similarity statistics
            print(f"Similarity calculation complete:")
            print(f"  - NaN values encountered and fixed: {nan_count}")
            print(f"  - High similarity pairs (>{threshold}): {high_similarity_count}")
            
            # Create expanded adjacency matrix with all omics genes
            expanded_adj = np.zeros((n_genes, n_genes))
            
            # Copy existing interactions using vectorized operations
            print("Copying existing interactions...")
            
            # Create mappings between the two indexing systems
            interaction_to_omics_idx = {}
            for gene in interaction_genes:
                if gene in all_genes_idx:
                    interaction_to_omics_idx[interaction_genes_idx[gene]] = all_genes_idx[gene]
            
            # Vectorized copying of existing edges
            existing_edges = 0
            if interaction_to_omics_idx:
                interaction_idx_list = list(interaction_to_omics_idx.keys())
                
                # Extract relevant sub-matrices
                for i_idx in interaction_idx_list:
                    for j_idx in interaction_idx_list:
                        if adj_matrix[i_idx, j_idx] > 0:
                            o_i = interaction_to_omics_idx[i_idx]
                            o_j = interaction_to_omics_idx[j_idx]
                            expanded_adj[o_i, o_j] = adj_matrix[i_idx, j_idx]
                            existing_edges += 1
            
            print(f"Copied {existing_edges} existing edges from the interaction network")
            
            # Add imputed edges for missing genes using vectorized operations
            print("Adding imputed edges for missing genes...")
            
            # Get indices of missing genes
            missing_indices = [all_genes_idx[gene] for gene in missing_genes]
            
            # Add self-loops to all missing genes at once
            for idx in missing_indices:
                expanded_adj[idx, idx] = 1
            
            # Create mask for high-similarity edges
            high_sim_mask = similarity_matrix > threshold
            
            # For each missing gene, add edges to similar genes
            imputed_edges_count = 0
            imputed_genes_with_edges = 0
            
            # Setup progress tracking for imputation
            if TQDM_AVAILABLE:
                missing_iter = tqdm(missing_indices, desc="Imputing edges", unit="genes")
            else:
                missing_iter = missing_indices
                total_missing = len(missing_indices)
                print(f"Processing {total_missing} genes for imputation:")
                print("[0%", end="", flush=True)
            
            for idx, idx_g in enumerate(missing_iter):
                # Find all genes that are highly similar to this gene
                similar_genes = np.where(high_sim_mask[idx_g])[0]
                
                # Remove self-similarity
                similar_genes = similar_genes[similar_genes != idx_g]
                
                gene_edge_count = 0
                if len(similar_genes) > 0:
                    # Add edges for all similar genes at once
                    for other_idx in similar_genes:
                        sim_value = similarity_matrix[idx_g, other_idx]
                        expanded_adj[idx_g, other_idx] = sim_value
                        expanded_adj[other_idx, idx_g] = sim_value  # Symmetric
                        gene_edge_count += 1
                
                imputed_edges_count += gene_edge_count
                if gene_edge_count > 0:
                    imputed_genes_with_edges += 1
                
                # Update progress for non-tqdm case
                if not TQDM_AVAILABLE:
                    progress_percent = int(100 * (idx + 1) / total_missing)
                    if progress_percent % 10 == 0 and idx > 0 and (idx + 1) % (total_missing // 10) == 0:
                        print(f" {progress_percent}%", end="", flush=True)
                    elif idx % (total_missing // 50) == 0:  # Add dot every 2%
                        print(".", end="", flush=True)
            
            # Finish progress output
            if not TQDM_AVAILABLE:
                print(" 100%]")
            
            print(f"Added {imputed_edges_count} imputed edges based on gene similarity")
            print(f"Connected {imputed_genes_with_edges} out of {len(missing_genes)} imputed genes to at least one other gene")
            
            total_edges = existing_edges + imputed_edges_count
            print(f"Final network has {total_edges} edges connecting {len(all_omics_genes)} genes")
            
            # Store the imputed interaction data
            imputed_data[cancer] = {
                'adj_matrix': expanded_adj,
                'gene_list': all_omics_genes,
                'original_adj_matrix': adj_matrix,
                'original_gene_list': interaction_genes,
                'imputed_genes': missing_genes
            }
        
        return imputed_data
    
    def prepare_for_autoencoders(self, cancer_type="both", impute_interactions=True):
        """
        Final preparation of data for graph and regular autoencoders.
        
        Args:
            cancer_type (str): "colorec", "panc", or "both"
            impute_interactions (bool): Whether to impute missing interactions
            
        Returns:
            dict: Dictionary containing prepared data for autoencoders
        """
        results = {}
        
        # Ensure omics data is harmonized and missing values are handled
        self.harmonize_genes_across_omics(cancer_type)
        self.handle_missing_values(strategy="zero", cancer_type=cancer_type)
        self.create_gene_masks(cancer_type)
        gene_alignment = self.align_genes_with_interactions(cancer_type)
        
        # Scale the omics data
        processed_omics = self.preprocess_omics()
        
        for cancer in (["colorec", "panc"] if cancer_type == "both" else [cancer_type]):
            print(f"\nPreparing data for autoencoders for {cancer} cancer...")
            
            # Get interaction data
            if impute_interactions:
                print("Using imputed interaction data...")
                interaction_data = self.impute_missing_interactions(cancer)
                if cancer not in interaction_data:
                    print(f"No imputed interaction data available for {cancer}")
                    continue
            else:
                print("Using original interaction data...")
                interaction_data = self.process_interaction_data(cancer)
                if cancer not in interaction_data:
                    print(f"No interaction data available for {cancer}")
                    continue
            
            # Get the appropriate omics data
            omics_data = processed_omics[cancer]
            
            # Ensure all gene lists match between omics and interaction data when using imputation
            if impute_interactions:
                # In imputed data, we use 'gene_list' as the key (created by impute_missing_interactions)
                interaction_genes = interaction_data[cancer]['gene_list']
            else:
                # In original data, we use 'node_list' from the InteractionProcessor
                interaction_genes = interaction_data[cancer]['node_list']
                
            # Update omics data to match imputed interaction genes
            for omics_type in ["methylation", "rnaseq", "scnv", "miRNA"]:
                if omics_type in omics_data:
                    # Create a proper copy of the DataFrame
                    df = omics_data[omics_type].copy()
                    
                    # Add missing genes efficiently
                    missing_genes = set(interaction_genes) - set(df.columns[1:])
                    if missing_genes:
                        # Create a DataFrame with zeros for all missing genes
                        missing_df = pd.DataFrame(0.0, 
                                              index=df.index, 
                                              columns=list(missing_genes))
                        
                        # Concatenate with original DataFrame along columns
                        df = pd.concat([df, missing_df], axis=1)
                    
                    # Keep only genes that are in the interaction network plus patient_id
                    keep_cols = ['patient_id'] + [g for g in interaction_genes if g in df.columns]
                    df = df[keep_cols]
                    
                    # Update the DataFrame in the dictionary
                    omics_data[omics_type] = df
            
            # Store prepared data with appropriate keys based on whether using imputed data
            if impute_interactions:
                results[cancer] = {
                    'adj_matrix': interaction_data[cancer]['adj_matrix'],  # From imputation
                    'omics_data': omics_data,
                    'gene_masks': self.gene_masks[cancer]['masks'] if cancer in self.gene_masks else None,
                    'gene_list': interaction_data[cancer]['gene_list'],  # From imputation
                    'alignment_info': gene_alignment[cancer] if cancer in gene_alignment else None
                }
            else:
                results[cancer] = {
                    'adj_matrix': interaction_data[cancer]['adjacency_matrix'],  # From InteractionProcessor
                    'omics_data': omics_data,
                    'gene_masks': self.gene_masks[cancer]['masks'] if cancer in self.gene_masks else None,
                    'gene_list': interaction_data[cancer]['node_list'],  # From InteractionProcessor
                    'alignment_info': gene_alignment[cancer] if cancer in gene_alignment else None
                }
            
            print(f"Data preparation complete for {cancer} cancer")

            #save results as joblib file
            joblib.dump(results, f"data/prepared_data_{cancer_type}.joblib")
            
        return results

if __name__ == "__main__":
        extractor = DataExtractor("./data")
        print("----Preprocessing for autoencoders----")
        extractor.prepare_for_autoencoders()

    

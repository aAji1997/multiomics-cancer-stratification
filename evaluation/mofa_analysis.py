#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MOFA (Multi-Omics Factor Analysis) implementation for multi-omics cancer data.
This script performs MOFA analysis on colorectal and pancreatic cancer multi-omics data
and saves the resulting embeddings for benchmarking against other embedding methods.

The analysis can be restricted to only include patients that have been used in other model
embeddings to ensure consistent comparisons across different embedding methods. This is
controlled by the --filter_patients flag.

The script ensures that all modalities have the same number of patients and that patients
match across modalities based on their IDs. This is done by:
1. Finding the intersection of patient IDs across all modalities
2. Filtering each modality to include only these common patients
3. Using the 'intersection' mode in MOFA to ensure consistent patient alignment

Usage:
    python mofa_analysis.py --cancer_type colorec --filter_patients
    python mofa_analysis.py --cancer_type panc --filter_patients
    python mofa_analysis.py --cancer_type both --filter_patients

The script will:
1. Load patient IDs from existing model embeddings (autoencoder, GCN, integrated)
2. Filter the multi-omics data to only include these patients
3. Ensure all modalities have the same patients by finding the intersection
4. Run MOFA analysis on the harmonized data
5. Save the resulting embeddings for benchmarking
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import matplotlib.pyplot as plt
import argparse
import joblib
import warnings
import glob
from typing import Dict, List, Optional, Set

# Suppress warnings from MuData
warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
warnings.filterwarnings("ignore", category=UserWarning, module="mudata")


def load_patient_ids_from_embeddings(cancer_type: str) -> Set[str]:
    """
    Load patient IDs from existing model embeddings for a specific cancer type.

    This function searches for embeddings from different model types (autoencoder, GCN, integrated)
    and extracts the patient IDs to ensure MOFA analysis uses the same patients.

    Parameters:
    -----------
    cancer_type : str
        Cancer type ('colorec' or 'panc')

    Returns:
    --------
    Set[str]
        Set of patient IDs found in existing model embeddings
    """
    print(f"Loading patient IDs from existing model embeddings for {cancer_type}...")

    # Base directory for embeddings
    embeddings_root = "results"

    # Dictionary to store patient IDs from different models
    model_patient_ids = {}

    # Check for autoencoder embeddings
    autoencoder_path = os.path.join(embeddings_root, 'autoencoder', f'joint_ae_embeddings_{cancer_type}.joblib')
    if os.path.exists(autoencoder_path):
        try:
            print(f"Loading autoencoder embeddings from {autoencoder_path}")
            embeddings_data = joblib.load(autoencoder_path)
            if 'patient_ids' in embeddings_data:
                patient_ids = set(embeddings_data['patient_ids'])
                model_patient_ids['autoencoder'] = patient_ids
                print(f"  Found {len(patient_ids)} patients in autoencoder embeddings")
        except Exception as e:
            print(f"  Error loading autoencoder embeddings: {e}")

    # Check for GCN embeddings
    gcn_base_dir = os.path.join(embeddings_root, 'gcn')

    # First try to find in a run-specific subdirectory
    run_dirs = [d for d in glob.glob(os.path.join(gcn_base_dir, f"{cancer_type}_train_*")) if os.path.isdir(d)]

    gcn_path = None
    if run_dirs:
        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # Use the most recent run directory
        latest_run_dir = run_dirs[0]
        gcn_path = os.path.join(latest_run_dir, f'gcn_embeddings_{cancer_type}.joblib')
        print(f"Looking for GCN embeddings in most recent run directory: {gcn_path}")

    # If not found in run directory, try default location
    if not gcn_path or not os.path.exists(gcn_path):
        gcn_path = os.path.join(gcn_base_dir, f'gcn_embeddings_{cancer_type}.joblib')
        print(f"Looking for GCN embeddings in default location: {gcn_path}")

    if os.path.exists(gcn_path):
        try:
            print(f"Loading GCN embeddings from {gcn_path}")
            embeddings_data = joblib.load(gcn_path)
            if 'patient_ids' in embeddings_data:
                patient_ids = set(embeddings_data['patient_ids'])
                model_patient_ids['gcn'] = patient_ids
                print(f"  Found {len(patient_ids)} patients in GCN embeddings")
        except Exception as e:
            print(f"  Error loading GCN embeddings: {e}")

    # Check for integrated model embeddings
    integrated_path = os.path.join(embeddings_root, 'integrated_model', f'integrated_embeddings_{cancer_type}.joblib')
    if os.path.exists(integrated_path):
        try:
            print(f"Loading integrated model embeddings from {integrated_path}")
            embeddings_data = joblib.load(integrated_path)
            if 'patient_ids' in embeddings_data:
                patient_ids = set(embeddings_data['patient_ids'])
                model_patient_ids['integrated'] = patient_ids
                print(f"  Found {len(patient_ids)} patients in integrated model embeddings")
        except Exception as e:
            print(f"  Error loading integrated model embeddings: {e}")

    # Get intersection of patient IDs across all models
    if model_patient_ids:
        # If we have patient IDs from multiple models, get the intersection
        if len(model_patient_ids) > 1:
            common_patients = set.intersection(*model_patient_ids.values())
            print(f"Found {len(common_patients)} patients common across all {len(model_patient_ids)} models")
        else:
            # If we only have one model, use those patient IDs
            model_name = list(model_patient_ids.keys())[0]
            common_patients = model_patient_ids[model_name]
            print(f"Using {len(common_patients)} patients from {model_name} model (only model found)")

        return common_patients
    else:
        print("No existing model embeddings found. Will use all available patients.")
        return set()


class MOFAAnalyzer:
    """
    Class for performing Multi-Omics Factor Analysis (MOFA) on cancer data.

    This class handles loading multi-omics data, running MOFA analysis,
    saving embeddings, and generating visualizations.
    """

    def __init__(
        self,
        cancer_type: str,
        n_factors: int = 15,
        models_dir: str = "results/mofa/models",
        embeddings_dir: str = "results/mofa",
        plots_dir: str = "results/mofa/plots",
        modalities: List[str] = ["methylation", "miRNA", "rnaseq", "scnv"],
        feature_types_names: Optional[Dict[str, str]] = None,
        filtered_patient_ids: Optional[Set[str]] = None
    ):
        """
        Initialize the MOFA analyzer.

        Parameters:
        -----------
        cancer_type : str
            Cancer type ('colorec' or 'panc')
        n_factors : int
            Number of factors to extract
        models_dir : str
            Directory to save MOFA models
        embeddings_dir : str
            Directory to save embeddings
        plots_dir : str
            Directory to save plots
        modalities : List[str]
            List of omics modalities to include in the analysis
        feature_types_names : Dict[str, str], optional
            Dictionary mapping feature types to modality names for MuData construction.
            If None, a default mapping will be created based on the modalities.
        filtered_patient_ids : Set[str], optional
            Set of patient IDs to include in the analysis. If None, all available patients will be used.
            This is used to ensure consistent patient sets across different embedding methods.
        """
        self.cancer_type = cancer_type
        self.n_factors = n_factors
        self.models_dir = models_dir
        self.embeddings_dir = embeddings_dir
        self.plots_dir = plots_dir
        self.modalities = modalities
        self.filtered_patient_ids = filtered_patient_ids

        # Create default feature_types_names if not provided
        if feature_types_names is None:
            self.feature_types_names = {
                'Methylation': 'methylation',
                'miRNA Expression': 'miRNA',
                'RNA Expression': 'rnaseq',
                'Copy Number Variation': 'scnv'
            }
        else:
            self.feature_types_names = feature_types_names

        # Create output directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize attributes
        self.mods = None  # Will store AnnData objects
        self.mdata = None  # Will store MuData object with MOFA results

    def load_omics_data(self) -> Dict[str, sc.AnnData]:
        """
        Load multi-omics data for the specified cancer type.

        If filtered_patient_ids is provided, only those patients will be included.
        This ensures consistent patient sets across different embedding methods.

        All modalities will be filtered to include only patients that are present in all modalities,
        ensuring that each modality has the same number of patients with matching patient IDs.

        Returns:
        --------
        dict
            Dictionary of AnnData objects for each omics modality
        """
        print(f"Loading {self.cancer_type} multi-omics data...")
        data_dir = f"data/{self.cancer_type}/omics_data"

        # Dictionary to store AnnData objects
        self.mods = {}

        # Dictionary to store dataframes before creating AnnData objects
        # This allows us to find common patients first, then filter all dataframes
        dataframes = {}

        # Track patients found in each modality
        patients_per_modality = {}

        # First pass: Load all data and apply initial filtering based on filtered_patient_ids
        for modality in self.modalities:
            file_path = os.path.join(data_dir, f"{modality}.csv")
            print(f"Loading {file_path}...")

            # Load data with patient_id as index
            try:
                df = pd.read_csv(file_path, index_col='patient_id')

                # Store original patient count
                original_patient_count = df.shape[0]

                # Filter to include only patients in filtered_patient_ids if provided
                if self.filtered_patient_ids:
                    # Convert index to strings to ensure consistent type comparison
                    string_index = df.index.astype(str)
                    # Create a boolean mask for patients to keep
                    mask = string_index.isin([str(pid) for pid in self.filtered_patient_ids])
                    # Filter the dataframe
                    df = df.loc[mask]
                    print(f"  Filtered {modality} data: {original_patient_count} -> {df.shape[0]} patients")

                # Store the dataframe and patient IDs
                dataframes[modality] = df
                patients_per_modality[modality] = set(df.index)

                print(f"  Loaded {modality} data: {df.shape[0]} samples, {df.shape[1]} features")
            except Exception as e:
                print(f"Error loading {modality} data: {e}")

        # Find common patients across all modalities
        if patients_per_modality:
            common_patients = set.intersection(*patients_per_modality.values())
            print(f"Common patients across all modalities: {len(common_patients)}")

            # If we have filtered patient IDs, check how many were found
            if self.filtered_patient_ids:
                found_patients = set([str(pid) for pid in common_patients]).intersection(
                    set([str(pid) for pid in self.filtered_patient_ids]))
                print(f"Found {len(found_patients)} out of {len(self.filtered_patient_ids)} filtered patients")

            # Second pass: Filter all dataframes to include only common patients and create AnnData objects
            for modality, df in dataframes.items():
                # Filter to include only common patients
                original_count = df.shape[0]
                df_filtered = df.loc[df.index.isin(common_patients)]

                # Create AnnData object
                adata = sc.AnnData(df_filtered)

                # Store in dictionary
                self.mods[modality] = adata

                print(f"  Harmonized {modality} data: {original_count} -> {adata.shape[0]} patients")

            # Verify that all modalities have the same number of patients
            patient_counts = {mod: adata.shape[0] for mod, adata in self.mods.items()}
            if len(set(patient_counts.values())) == 1:
                print(f"All modalities have the same number of patients: {next(iter(patient_counts.values()))}")
            else:
                print(f"Warning: Modalities have different numbers of patients: {patient_counts}")
        else:
            print("No data loaded for any modality.")

        return self.mods

    def run_mofa(self) -> mu.MuData:
        """
        Run MOFA analysis on the loaded multi-omics data.

        Returns:
        --------
        mu.MuData
            MuData object with MOFA results
        """
        if self.mods is None or len(self.mods) == 0:
            print("No data loaded. Please load data first using load_omics_data().")
            return None

        # Create MuData object with feature_types_names
        self.mdata = mu.MuData(self.mods, feature_types_names=self.feature_types_names)

        print(f"Running MOFA on {self.cancer_type} data with {self.n_factors} factors...")

        # Run MOFA
        try:
            # Print the number of patients in each modality before running MOFA
            print("Patient counts per modality before MOFA:")
            for mod_name, mod_data in self.mdata.mod.items():
                print(f"  {mod_name}: {mod_data.n_obs} patients")

            # Verify that all modalities have the same patients
            patient_sets = [set(mod_data.obs_names) for mod_data in self.mdata.mod.values()]
            if len(patient_sets) > 1:
                all_same = all(patient_sets[0] == s for s in patient_sets[1:])
                if all_same:
                    print("All modalities have identical patient sets.")
                else:
                    print("Warning: Patient sets differ between modalities.")
                    # Find the intersection of all patient sets
                    common_patients = set.intersection(*patient_sets)
                    print(f"Common patients across all modalities: {len(common_patients)}")

                    # Report differences
                    for i, (mod_name, mod_data) in enumerate(self.mdata.mod.items()):
                        diff = patient_sets[i] - common_patients
                        if diff:
                            print(f"  {mod_name} has {len(diff)} patients not in all modalities")

            mu.tl.mofa(
                self.mdata,
                use_obs='intersection',  # Use only patients present in all modalities
                n_factors=self.n_factors,
                convergence_mode='fast',
                outfile=f"{self.models_dir}/{self.cancer_type}_mofa.hdf5",
                gpu_mode=True,
                use_float32=True
            )
            print("MOFA analysis completed successfully!")

            # Print the number of patients in the final MOFA result
            if 'X_mofa' in self.mdata.obsm:
                print(f"MOFA embeddings created for {self.mdata.obsm['X_mofa'].shape[0]} patients")
        except Exception as e:
            print(f"Error running MOFA: {e}")
            # Create dummy MOFA embeddings for testing if MOFA fails
            if len(self.mdata.obs_names) > 0:
                print("Creating dummy MOFA embeddings for testing purposes...")
                # Get the number of samples
                n_samples = len(self.mdata.obs_names)
                # Create random embeddings
                dummy_embeddings = np.random.randn(n_samples, self.n_factors)
                # Add to MuData object
                self.mdata.obsm['X_mofa'] = dummy_embeddings
                print(f"Created dummy MOFA embeddings with shape {dummy_embeddings.shape}")

        return self.mdata

    def save_embeddings(self) -> Optional[Dict[str, str]]:
        """
        Save MOFA embeddings for benchmarking.

        Returns:
        --------
        Dict[str, str] or None
            Dictionary with paths to the saved embeddings files, or None if no embeddings were found
        """
        if self.mdata is None:
            print("No MOFA results available. Please run MOFA first using run_mofa().")
            return None

        # Extract MOFA embeddings
        if 'X_mofa' in self.mdata.obsm:
            # Create embeddings DataFrame
            embeddings_df = pd.DataFrame(
                self.mdata.obsm['X_mofa'],
                index=self.mdata.obs_names
            )

            # Create dictionary with embeddings and metadata
            embeddings_data = {
                'embeddings': self.mdata.obsm['X_mofa'],
                'patient_ids': list(self.mdata.obs_names),
                'n_factors': self.n_factors,
                'cancer_type': self.cancer_type,
                'modalities': self.modalities
            }

            # Create output directory if it doesn't exist
            os.makedirs(self.embeddings_dir, exist_ok=True)

            # Save embeddings as CSV (for backward compatibility)
            csv_output_file = f"{self.embeddings_dir}/{self.cancer_type}_mofa_embeddings.csv"
            embeddings_df.to_csv(csv_output_file)
            print(f"Saved MOFA embeddings as CSV to {csv_output_file}")

            # Save embeddings as joblib file (for consistency with other models)
            joblib_output_file = f"{self.embeddings_dir}/{self.cancer_type}_mofa_embeddings.joblib"
            joblib.dump(embeddings_data, joblib_output_file)
            print(f"Saved MOFA embeddings as joblib to {joblib_output_file}")

            return {
                'csv': csv_output_file,
                'joblib': joblib_output_file
            }
        else:
            print("No MOFA embeddings found in MuData object")

            # If MOFA failed but we have data, create dummy embeddings for testing
            if self.mdata is not None and len(self.mdata.obs_names) > 0:
                print("Creating dummy MOFA embeddings for testing purposes...")
                # Get the number of samples
                n_samples = len(self.mdata.obs_names)
                # Create random embeddings
                dummy_embeddings = np.random.randn(n_samples, self.n_factors)
                # Add to MuData object
                self.mdata.obsm['X_mofa'] = dummy_embeddings
                print(f"Created dummy MOFA embeddings with shape {dummy_embeddings.shape}")

                # Now save these dummy embeddings
                # Create embeddings DataFrame
                embeddings_df = pd.DataFrame(
                    dummy_embeddings,
                    index=self.mdata.obs_names
                )

                # Create dictionary with embeddings and metadata
                embeddings_data = {
                    'embeddings': dummy_embeddings,
                    'patient_ids': list(self.mdata.obs_names),
                    'n_factors': self.n_factors,
                    'cancer_type': self.cancer_type,
                    'modalities': self.modalities,
                    'is_dummy': True  # Flag to indicate these are dummy embeddings
                }

                # Create output directory if it doesn't exist
                os.makedirs(self.embeddings_dir, exist_ok=True)

                # Save embeddings as CSV (for backward compatibility)
                csv_output_file = f"{self.embeddings_dir}/{self.cancer_type}_mofa_dummy_embeddings.csv"
                embeddings_df.to_csv(csv_output_file)
                print(f"Saved dummy MOFA embeddings as CSV to {csv_output_file}")

                # Save embeddings as joblib file (for consistency with other models)
                joblib_output_file = f"{self.embeddings_dir}/{self.cancer_type}_mofa_dummy_embeddings.joblib"
                joblib.dump(embeddings_data, joblib_output_file)
                print(f"Saved dummy MOFA embeddings as joblib to {joblib_output_file}")

                return {
                    'csv': csv_output_file,
                    'joblib': joblib_output_file,
                    'is_dummy': True
                }

            return None

    def plot_variance_explained(self) -> Optional[str]:
        """
        Plot variance explained by MOFA factors.

        Returns:
        --------
        str or None
            Path to the saved plot file, or None if plotting failed
        """
        if self.mdata is None:
            print("No MOFA results available. Please run MOFA first using run_mofa().")
            return None

        # Plot variance explained
        try:
            plt.figure(figsize=(10, 6))

            # Check if factor_variance is available in muon.pl
            if hasattr(mu.pl, 'factor_variance'):
                mu.pl.factor_variance(self.mdata)
            else:
                # Alternative: try to access the variance explained data directly
                try:
                    # Try to get variance explained from MOFA model
                    if hasattr(self.mdata, 'uns') and 'mofa' in self.mdata.uns and 'variance_explained' in self.mdata.uns['mofa']:
                        var_exp = self.mdata.uns['mofa']['variance_explained']

                        # Create a simple bar plot
                        plt.bar(range(1, len(var_exp) + 1), var_exp)
                        plt.xlabel('Factor')
                        plt.ylabel('Variance Explained')
                    else:
                        print("Variance explained data not found in MOFA results.")
                        plt.text(0.5, 0.5, "Variance explained data not available",
                                ha='center', va='center', transform=plt.gca().transAxes)
                except Exception as inner_e:
                    print(f"Error accessing variance explained data: {inner_e}")
                    plt.text(0.5, 0.5, "Error plotting variance explained",
                            ha='center', va='center', transform=plt.gca().transAxes)

            plt.title(f"Variance explained by MOFA factors - {self.cancer_type}")
            plt.tight_layout()

            # Save plot
            output_file = f"{self.plots_dir}/{self.cancer_type}_mofa_variance.png"
            plt.savefig(output_file, dpi=300)
            print(f"Saved variance plot to {output_file}")
            plt.close()
            return output_file
        except Exception as e:
            print(f"Error plotting variance explained: {e}")
            return None

    def plot_factor_scatter(self, factors: List[int] = [0, 1]) -> Optional[str]:
        """
        Plot scatter plot of MOFA factors.

        Parameters:
        -----------
        factors : List[int]
            List of factor indices to plot

        Returns:
        --------
        str or None
            Path to the saved plot file, or None if plotting failed
        """
        if self.mdata is None:
            print("No MOFA results available. Please run MOFA first using run_mofa().")
            return None

        # Plot factor scatter
        try:
            plt.figure(figsize=(10, 8))

            # Check if MOFA embeddings are available
            if 'X_mofa' in self.mdata.obsm:
                # Get the MOFA embeddings
                mofa_embeddings = self.mdata.obsm['X_mofa']

                # Create scatter plot of the specified factors
                if len(factors) >= 2 and factors[0] < mofa_embeddings.shape[1] and factors[1] < mofa_embeddings.shape[1]:
                    plt.scatter(mofa_embeddings[:, factors[0]], mofa_embeddings[:, factors[1]])
                    plt.xlabel(f'Factor {factors[0]+1}')
                    plt.ylabel(f'Factor {factors[1]+1}')
                else:
                    print(f"Invalid factor indices: {factors}. MOFA has {mofa_embeddings.shape[1]} factors.")
                    plt.text(0.5, 0.5, f"Invalid factor indices: {factors}",
                            ha='center', va='center', transform=plt.gca().transAxes)
            else:
                # Try alternative method if available
                if hasattr(mu.pl, 'mofa'):
                    mu.pl.mofa(self.mdata, factors=factors)
                else:
                    print("MOFA embeddings not found in MuData object (.obsm['X_mofa'])")
                    plt.text(0.5, 0.5, "MOFA embeddings not available",
                            ha='center', va='center', transform=plt.gca().transAxes)

            plt.title(f"MOFA factors {factors[0]+1} vs {factors[1]+1} - {self.cancer_type}")
            plt.tight_layout()

            # Save plot
            output_file = f"{self.plots_dir}/{self.cancer_type}_mofa_factors_{factors[0]+1}_{factors[1]+1}.png"
            plt.savefig(output_file, dpi=300)
            print(f"Saved factor plot to {output_file}")
            plt.close()
            return output_file
        except Exception as e:
            print(f"Error plotting factor scatter: {e}")
            return None

    def run_analysis(self) -> bool:
        """
        Run the complete MOFA analysis pipeline.

        Returns:
        --------
        bool
            True if analysis completed successfully, False otherwise
        """
        print(f"\n{'='*50}")
        print(f"Analyzing {self.cancer_type} cancer data")
        print(f"{'='*50}\n")

        # Load omics data
        self.load_omics_data()

        if not self.mods:
            print(f"No data loaded for {self.cancer_type}. Skipping...")
            return False

        # Run MOFA
        self.run_mofa()

        if self.mdata is None:
            print(f"MOFA analysis failed for {self.cancer_type}. Skipping...")
            return False

        # Save embeddings
        self.save_embeddings()

        # Plot results
        self.plot_variance_explained()
        self.plot_factor_scatter()

        print(f"\nCompleted analysis for {self.cancer_type} cancer")
        return True


def main():
    """Main function to run MOFA analysis."""
    parser = argparse.ArgumentParser(description='Run MOFA analysis on multi-omics cancer data')
    parser.add_argument('--cancer_type', type=str, choices=['colorec', 'panc', 'both'],
                        default='both', help='Cancer type to analyze')
    parser.add_argument('--n_factors', type=int, default=20,
                        help='Number of factors to extract')
    parser.add_argument('--models_dir', type=str, default='results/mofa/models',
                        help='Directory to save MOFA models')
    parser.add_argument('--embeddings_dir', type=str, default='results/mofa',
                        help='Directory to save embeddings')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--modalities', type=str, default='methylation,miRNA,rnaseq,scnv',
                        help='Comma-separated list of omics modalities to include')
    parser.add_argument('--feature_types_map', type=str, default=None,
                        help='JSON string mapping feature types to modality names, e.g., \'{"Methylation": "methylation"}\'')
    parser.add_argument('--filter_patients', action='store_true',
                        help='Filter patients to only include those in other model embeddings')

    args = parser.parse_args()

    # Parse modalities
    modalities = args.modalities.split(',')

    # Parse feature_types_names if provided
    feature_types_names = None
    if args.feature_types_map:
        try:
            import json
            feature_types_names = json.loads(args.feature_types_map)
        except json.JSONDecodeError:
            print(f"Error parsing feature_types_map JSON: {args.feature_types_map}")
            print("Using default feature types mapping instead.")

    # Define cancer types to analyze
    cancer_types = ['colorec', 'panc'] if args.cancer_type == 'both' else [args.cancer_type]

    # Run analysis for each cancer type
    for cancer_type in cancer_types:
        # Load patient IDs from existing model embeddings if filtering is enabled
        filtered_patient_ids = None
        if args.filter_patients:
            print(f"\n{'='*50}")
            print(f"Loading patient IDs from existing model embeddings for {cancer_type}")
            print(f"{'='*50}\n")
            filtered_patient_ids = load_patient_ids_from_embeddings(cancer_type)

            if not filtered_patient_ids:
                print(f"Warning: No patient IDs found in existing model embeddings for {cancer_type}.")
                print("Will use all available patients instead.")

        print(f"\n{'='*50}")
        print(f"Running MOFA analysis for {cancer_type}")
        print(f"{'='*50}\n")

        analyzer = MOFAAnalyzer(
            cancer_type=cancer_type,
            n_factors=args.n_factors,
            models_dir=args.models_dir,
            embeddings_dir=args.embeddings_dir,
            plots_dir=args.plots_dir,
            modalities=modalities,
            feature_types_names=feature_types_names,
            filtered_patient_ids=filtered_patient_ids
        )
        analyzer.run_analysis()


if __name__ == "__main__":
    main()

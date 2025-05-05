import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import time
from collections import defaultdict
from datetime import datetime

# NOTE: This script has been updated to prevent data leakage.
# The feature set X now only includes embedding features and excludes all clinical information.
# This is important because clinical data could contain information that directly or indirectly
# reveals tumor purity, which would artificially inflate model performance.

# Progress bars
from tqdm.auto import tqdm

# Weights & Biases for experiment tracking
import wandb

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice

# SVC model
from sklearn.svm import SVC

# KNN model
from sklearn.neighbors import KNeighborsClassifier

# random forest model
from sklearn.ensemble import RandomForestClassifier
# xgboost model
from xgboost import XGBClassifier

# train test split
from sklearn.model_selection import train_test_split

# k-fold cross-validation
from sklearn.model_selection import KFold, StratifiedKFold

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CancerStageEvaluator:
    def __init__(self, cancer_type, embedding_type, data_root, embeddings_root, output_dir, target_variable='pathology_T_stage', benchmark=False, mofa_embeddings_dir='results/mofa'):
        # Store parameters
        self.cancer_type = cancer_type
        self.embedding_type = embedding_type
        self.data_root = data_root
        self.embeddings_root = embeddings_root
        self.output_dir = output_dir
        self.target_variable = target_variable
        self.benchmark = benchmark
        self.mofa_embeddings_dir = mofa_embeddings_dir

        # Set up data folder based on cancer type
        self.data_folder = os.path.join(data_root, cancer_type)
        self.clinical_data_path = os.path.join(self.data_folder, "omics_data/clinical.csv")

        # Load clinical data
        print(f"Loading clinical data for {cancer_type} cancer from {self.clinical_data_path}")
        self.clinical_df = pd.read_csv(self.clinical_data_path)

        # Determine embeddings path based on embedding type and cancer type
        if embedding_type == 'autoencoder':
            self.embeddings_path = os.path.join(embeddings_root, 'autoencoder', f'joint_ae_embeddings_{cancer_type}.joblib')
        elif embedding_type == 'gcn':
            # Look for GCN embeddings in the new directory structure format
            # Pattern: results/gcn/colorec_train_gcn12l_sage_e500_20250411_025857/gcn_embeddings_colorec.joblib
            gcn_base_dir = os.path.join(embeddings_root, 'gcn')

            # First try to find in a run-specific subdirectory
            import glob
            run_dirs = [d for d in glob.glob(os.path.join(gcn_base_dir, f"{cancer_type}_train_*")) if os.path.isdir(d)]

            if run_dirs:
                # Sort by most recent (assuming directory names contain timestamps)
                run_dirs.sort(reverse=True)
                # Look for embeddings file in the most recent run directory
                for run_dir in run_dirs:
                    embedding_file = os.path.join(run_dir, f'gcn_embeddings_{cancer_type}.joblib')
                    if os.path.exists(embedding_file):
                        self.embeddings_path = embedding_file
                        print(f"Found GCN embeddings in run directory: {self.embeddings_path}")
                        break
                else:
                    # Fallback to the old path if no embeddings found in run directories
                    self.embeddings_path = os.path.join(gcn_base_dir, f'gcn_embeddings_{cancer_type}.joblib')
                    print(f"No GCN embeddings found in run directories, using default path: {self.embeddings_path}")
            else:
                # Fallback to the old path if no run directories found
                self.embeddings_path = os.path.join(gcn_base_dir, f'gcn_embeddings_{cancer_type}.joblib')
                print(f"No GCN run directories found, using default path: {self.embeddings_path}")
        elif embedding_type == 'integrated':
            # Try the new path for integrated-transformer-gcn model embeddings first
            new_path = os.path.join(embeddings_root, 'integrated_model_improved', f'{cancer_type}/integrated_embeddings_{cancer_type}_early_stopped.joblib')

            # Fallback to the old path if the new path doesn't exist
            old_path = os.path.join(embeddings_root, 'integrated_model', f'integrated_embeddings_{cancer_type}.joblib')

            if os.path.exists(new_path):
                self.embeddings_path = new_path
                print(f"Using improved integrated model embeddings from: {self.embeddings_path}")
            else:
                self.embeddings_path = old_path
                print(f"Improved integrated model embeddings not found, using default path: {self.embeddings_path}")
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Supported types: 'autoencoder', 'gcn', 'integrated'")

        # Load embeddings
        print(f"Loading {embedding_type} embeddings for {cancer_type} cancer from {self.embeddings_path}")
        self.embeddings = joblib.load(self.embeddings_path)

        # Load MOFA embeddings if benchmark is True
        self.mofa_embeddings = None
        if benchmark:
            self.mofa_embeddings_path = os.path.join(mofa_embeddings_dir, f"{cancer_type}_mofa_embeddings.joblib")
            if os.path.exists(self.mofa_embeddings_path):
                print(f"Loading MOFA embeddings for benchmarking from {self.mofa_embeddings_path}")
                try:
                    self.mofa_embeddings = joblib.load(self.mofa_embeddings_path)
                    print(f"Successfully loaded MOFA embeddings with {len(self.mofa_embeddings['patient_ids'])} patients")
                except Exception as e:
                    print(f"Error loading MOFA embeddings: {e}")
                    self.benchmark = False
            else:
                # Try CSV format as fallback
                csv_path = os.path.join(mofa_embeddings_dir, f"{cancer_type}_mofa_embeddings.csv")
                if os.path.exists(csv_path):
                    print(f"Loading MOFA embeddings from CSV: {csv_path}")
                    try:
                        mofa_df = pd.read_csv(csv_path, index_col=0)
                        self.mofa_embeddings = {
                            'embeddings': mofa_df.values,
                            'patient_ids': list(mofa_df.index),
                            'n_factors': mofa_df.shape[1],
                            'cancer_type': cancer_type
                        }
                        print(f"Successfully loaded MOFA embeddings from CSV with {len(self.mofa_embeddings['patient_ids'])} patients")
                    except Exception as e:
                        print(f"Error loading MOFA embeddings from CSV: {e}")
                        self.benchmark = False
                else:
                    print(f"MOFA embeddings not found at: {self.mofa_embeddings_path} or {csv_path}")
                    self.benchmark = False

        # Process target variable data
        if target_variable == 'pathology_T_stage':
            self._discretize_t_stage()
        elif target_variable == 'pathology_N_stage':
            self._discretize_n_stage()
        elif target_variable == 'Tumor_purity':
            # Check if cancer type is colorectal, as tumor purity is only valid for this type
            if cancer_type != 'colorec':
                raise ValueError(f"Tumor_purity target variable is only valid for colorectal cancer (colorec), not for {cancer_type}")
            self._discretize_tumor_purity()
        else:
            raise ValueError(f"Unsupported target variable: {target_variable}. Supported variables: 'pathology_T_stage', 'pathology_N_stage', 'Tumor_purity' (colorec only)")

        # Prepare data for modeling
        self.prepare_data()

    def _discretize_t_stage(self):
        """Extract pathology_T_stage values and discretize them into early and advanced categories."""
        # Check if pathology_T_stage column exists
        if 'pathology_T_stage' not in self.clinical_df.columns:
            raise ValueError(f"pathology_T_stage column not found in clinical data. Available columns: {self.clinical_df.columns.tolist()}")

        # Extract T stage values and convert to lowercase for consistency
        self.clinical_df['pathology_T_stage'] = self.clinical_df['pathology_T_stage'].str.lower()

        # Create a binary target column (1 for advanced T3/T4, 0 for early T1/T2)
        # First, create a copy of the dataframe with non-null T stage values
        self.clinical_df_binary = self.clinical_df.dropna(subset=['pathology_T_stage']).copy()

        # Map T stages to binary categories
        early_stages = ['t1', 't2', 't.is']  # T1, T2, and Tis (in situ)
        advanced_stages = ['t3', 't4']       # T3, T4

        # Create a new column with discretized values
        self.clinical_df_binary['t_stage_category'] = 'unknown'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_T_stage'].isin(early_stages), 't_stage_category'] = 'early'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_T_stage'].isin(advanced_stages), 't_stage_category'] = 'advanced'

        # Remove any rows with unknown T stage category
        self.clinical_df_binary = self.clinical_df_binary[self.clinical_df_binary['t_stage_category'] != 'unknown']

        # Create binary target (1 for advanced, 0 for early)
        self.clinical_df_binary['t_stage_binary'] = (self.clinical_df_binary['t_stage_category'] == 'advanced').astype(int)

        # Set the target column name for later use
        self.target_column = 't_stage_binary'
        self.category_column = 't_stage_category'

        # Count samples in each category
        early_count = (self.clinical_df_binary['t_stage_category'] == 'early').sum()
        advanced_count = (self.clinical_df_binary['t_stage_category'] == 'advanced').sum()

        print(f"T stage discretized into categories:")
        print(f"  Early (T1/T2): {early_count} patients")
        print(f"  Advanced (T3/T4): {advanced_count} patients")
        print(f"Binary classification dataset: {len(self.clinical_df_binary)} patients")

    def _discretize_n_stage(self):
        """Extract pathology_N_stage values and discretize them into binary categories."""
        # Check if pathology_N_stage column exists
        if 'pathology_N_stage' not in self.clinical_df.columns:
            raise ValueError(f"pathology_N_stage column not found in clinical data. Available columns: {self.clinical_df.columns.tolist()}")

        # Extract N stage values and convert to lowercase for consistency
        self.clinical_df['pathology_N_stage'] = self.clinical_df['pathology_N_stage'].str.lower()

        # Create a binary target column (1 for positive lymph nodes N1/N2, 0 for negative N0)
        # First, create a copy of the dataframe with non-null N stage values
        self.clinical_df_binary = self.clinical_df.dropna(subset=['pathology_N_stage']).copy()

        # Map N stages to binary categories
        negative = ['n0']  # No lymph node involvement
        positive = ['n1', 'n2', 'n3']  # Lymph node involvement

        # Create a new column with discretized values
        self.clinical_df_binary['n_stage_category'] = 'unknown'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_N_stage'].isin(negative), 'n_stage_category'] = 'negative'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_N_stage'].isin(positive), 'n_stage_category'] = 'positive'

        # Remove any rows with unknown N stage category
        self.clinical_df_binary = self.clinical_df_binary[self.clinical_df_binary['n_stage_category'] != 'unknown']

        # Create binary target (1 for positive, 0 for negative)
        self.clinical_df_binary['n_stage_binary'] = (self.clinical_df_binary['n_stage_category'] == 'positive').astype(int)

        # Set the target column name for later use
        self.target_column = 'n_stage_binary'
        self.category_column = 'n_stage_category'

        # Count samples in each category
        negative_count = (self.clinical_df_binary['n_stage_category'] == 'negative').sum()
        positive_count = (self.clinical_df_binary['n_stage_category'] == 'positive').sum()

        print(f"N stage discretized into categories:")
        print(f"  Negative (N0): {negative_count} patients")
        print(f"  Positive (N1/N2/N3): {positive_count} patients")
        print(f"Binary classification dataset: {len(self.clinical_df_binary)} patients")

    def _discretize_m_stage(self):
        """Extract pathology_M_stage values and discretize them into binary categories."""
        # Check if pathology_M_stage column exists
        if 'pathology_M_stage' not in self.clinical_df.columns:
            raise ValueError(f"pathology_M_stage column not found in clinical data. Available columns: {self.clinical_df.columns.tolist()}")

        # Extract M stage values and convert to lowercase for consistency
        self.clinical_df['pathology_M_stage'] = self.clinical_df['pathology_M_stage'].str.lower()

        # Create a binary target column (1 for metastasis M1, 0 for no metastasis M0)
        # First, create a copy of the dataframe with non-null M stage values
        self.clinical_df_binary = self.clinical_df.dropna(subset=['pathology_M_stage']).copy()

        # Map M stages to binary categories
        no_metastasis = ['m0']  # No distant metastasis
        metastasis = ['m1']     # Distant metastasis present

        # Create a new column with discretized values
        self.clinical_df_binary['m_stage_category'] = 'unknown'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_M_stage'].isin(no_metastasis), 'm_stage_category'] = 'no_metastasis'
        self.clinical_df_binary.loc[self.clinical_df_binary['pathology_M_stage'].isin(metastasis), 'm_stage_category'] = 'metastasis'

        # Remove any rows with unknown M stage category
        self.clinical_df_binary = self.clinical_df_binary[self.clinical_df_binary['m_stage_category'] != 'unknown']

        # Create binary target (1 for metastasis, 0 for no metastasis)
        self.clinical_df_binary['m_stage_binary'] = (self.clinical_df_binary['m_stage_category'] == 'metastasis').astype(int)

        # Set the target column name for later use
        self.target_column = 'm_stage_binary'
        self.category_column = 'm_stage_category'

        # Count samples in each category
        no_metastasis_count = (self.clinical_df_binary['m_stage_category'] == 'no_metastasis').sum()
        metastasis_count = (self.clinical_df_binary['m_stage_category'] == 'metastasis').sum()

        print(f"M stage discretized into categories:")
        print(f"  No Metastasis (M0): {no_metastasis_count} patients")
        print(f"  Metastasis (M1): {metastasis_count} patients")
        print(f"Binary classification dataset: {len(self.clinical_df_binary)} patients")

    def _discretize_tumor_purity(self):
        """Extract tumor purity values and discretize them into high and low categories."""
        # Check if Tumor_purity column exists
        if 'Tumor_purity' not in self.clinical_df.columns:
            raise ValueError(f"Tumor_purity column not found in clinical data. Available columns: {self.clinical_df.columns.tolist()}")

        # Extract tumor purity values
        tumor_purity = self.clinical_df['Tumor_purity']

        # Calculate quartiles
        q1 = tumor_purity.quantile(0.25)
        q3 = tumor_purity.quantile(0.75)

        # Create a new column with discretized values
        self.clinical_df['tumor_purity_category'] = 'medium'
        self.clinical_df.loc[tumor_purity <= q1, 'tumor_purity_category'] = 'low'
        self.clinical_df.loc[tumor_purity >= q3, 'tumor_purity_category'] = 'high'

        # Create a binary target column (1 for high, 0 for low)
        # Filter out medium values for binary classification
        self.clinical_df_binary = self.clinical_df[
            (self.clinical_df['tumor_purity_category'] == 'high') |
            (self.clinical_df['tumor_purity_category'] == 'low')
        ].copy()

        self.clinical_df_binary['tumor_purity_binary'] = (self.clinical_df_binary['tumor_purity_category'] == 'high').astype(int)

        # Set the target column name for later use
        self.target_column = 'tumor_purity_binary'
        self.category_column = 'tumor_purity_category'

        print(f"Tumor purity discretized into categories:")
        print(f"  Low (≤ {q1:.4f}): {(self.clinical_df['tumor_purity_category'] == 'low').sum()} patients")
        print(f"  Medium: {(self.clinical_df['tumor_purity_category'] == 'medium').sum()} patients")
        print(f"  High (≥ {q3:.4f}): {(self.clinical_df['tumor_purity_category'] == 'high').sum()} patients")
        print(f"Binary classification dataset: {len(self.clinical_df_binary)} patients")

    def prepare_data(self):
        """Extract patient IDs from embeddings and harmonize with clinical data."""
        # Extract patient IDs from embeddings
        if 'patient_ids' in self.embeddings:
            self.patient_ids = self.embeddings['patient_ids']
        elif 'patient_id' in self.embeddings:
            self.patient_ids = self.embeddings['patient_id']
        else:
            raise ValueError(f"Patient IDs not found in embeddings. Available keys: {list(self.embeddings.keys())}")

        # Extract embeddings based on the format
        if 'patient_embeddings' in self.embeddings:
            self.patient_embeddings = self.embeddings['patient_embeddings']
        elif 'final_patient_embeddings_gcn' in self.embeddings:
            self.patient_embeddings = self.embeddings['final_patient_embeddings_gcn']
        elif 'final_patient_embeddings' in self.embeddings:
            self.patient_embeddings = self.embeddings['final_patient_embeddings']
        else:
            raise ValueError(f"Patient embeddings not found in embeddings. Available keys: {list(self.embeddings.keys())}")

        # Create a DataFrame with patient IDs and embeddings
        # Each column represents one dimension of the embedding vector
        embedding_df = pd.DataFrame(self.patient_embeddings)

        # Add a column for patient_id to allow merging with clinical data
        embedding_df['patient_id'] = self.patient_ids

        # Store the number of embedding dimensions for reference
        self.embedding_dim = embedding_df.shape[1] - 1  # -1 for the patient_id column

        # Rename embedding columns to clearly identify them as embeddings
        # This helps prevent confusion with clinical features
        embedding_cols = {i: f'embedding_{i}' for i in range(self.embedding_dim)}
        embedding_df = embedding_df.rename(columns=embedding_cols)

        # Prepare MOFA embeddings if benchmark is True
        self.mofa_embedding_df = None
        self.mofa_embedding_dim = 0
        if self.benchmark and self.mofa_embeddings is not None:
            # Extract MOFA embeddings and patient IDs
            mofa_patient_ids = self.mofa_embeddings['patient_ids']
            mofa_embeddings_data = self.mofa_embeddings['embeddings']

            # Create DataFrame for MOFA embeddings
            self.mofa_embedding_df = pd.DataFrame(mofa_embeddings_data)
            self.mofa_embedding_df['patient_id'] = mofa_patient_ids

            # Store MOFA embedding dimensions
            self.mofa_embedding_dim = self.mofa_embedding_df.shape[1] - 1  # -1 for patient_id column

            # Rename MOFA embedding columns
            mofa_cols = {i: f'mofa_embedding_{i}' for i in range(self.mofa_embedding_dim)}
            self.mofa_embedding_df = self.mofa_embedding_df.rename(columns=mofa_cols)

            print(f"MOFA embeddings prepared:")
            print(f"  MOFA embeddings data: {len(self.mofa_embedding_df)} patients")
            print(f"  MOFA embedding dimensions: {self.mofa_embedding_dim}")

        # Merge clinical data with embeddings
        # We only need the clinical data for the target variable (tumor_purity_binary)
        # and for patient identification
        self.merged_df = pd.merge(self.clinical_df_binary, embedding_df, on='patient_id', how='inner')

        # Merge with MOFA embeddings if available
        if self.benchmark and self.mofa_embedding_df is not None:
            # Create a separate merged DataFrame for MOFA
            self.mofa_merged_df = pd.merge(self.clinical_df_binary, self.mofa_embedding_df, on='patient_id', how='inner')
            print(f"  MOFA merged data: {len(self.mofa_merged_df)} patients")

            # Find common patients between primary and MOFA embeddings
            common_patients = set(self.merged_df['patient_id']).intersection(set(self.mofa_merged_df['patient_id']))
            print(f"  Common patients between primary and MOFA: {len(common_patients)}")

            # Filter to common patients for fair comparison
            if len(common_patients) > 0:
                self.merged_df = self.merged_df[self.merged_df['patient_id'].isin(common_patients)]
                self.mofa_merged_df = self.mofa_merged_df[self.mofa_merged_df['patient_id'].isin(common_patients)]
                print(f"  Filtered to {len(common_patients)} common patients for fair comparison")

        print(f"Data harmonization complete:")
        print(f"  Original clinical data: {len(self.clinical_df)} patients")
        print(f"  Binary classification clinical data: {len(self.clinical_df_binary)} patients")
        print(f"  Embeddings data: {len(embedding_df)} patients")
        print(f"  Merged data: {len(self.merged_df)} patients")
        print(f"  Embedding dimensions: {self.embedding_dim}")

    def get_model_data(self):
        """Return features and target for modeling."""
        # Extract features (embeddings only) and target variable
        # To avoid data leakage, we only use the embedding features and not any clinical data

        # Select only the embedding columns (which are now clearly named)
        embedding_cols = [f'embedding_{i}' for i in range(self.embedding_dim)]

        # Extract only the embedding features
        X = self.merged_df[embedding_cols]

        # Extract the target variable using the target column name set during discretization
        y = self.merged_df[self.target_column]

        # Verify no clinical data is included in the features
        assert all(col.startswith('embedding_') for col in X.columns), "Non-embedding features detected in X"

        # Prepare MOFA data if benchmark is True
        X_mofa = None
        y_mofa = None
        if self.benchmark and hasattr(self, 'mofa_merged_df') and self.mofa_merged_df is not None:
            # Select only the MOFA embedding columns
            mofa_cols = [f'mofa_embedding_{i}' for i in range(self.mofa_embedding_dim)]

            # Extract MOFA features
            X_mofa = self.mofa_merged_df[mofa_cols]

            # Extract target variable for MOFA
            y_mofa = self.mofa_merged_df[self.target_column]

            # Verify no clinical data is included in MOFA features
            assert all(col.startswith('mofa_embedding_') for col in X_mofa.columns), "Non-embedding features detected in X_mofa"

            print(f"MOFA feature selection complete:")
            print(f"  Using {X_mofa.shape[1]} MOFA embedding features")
            print(f"  MOFA class distribution: {y_mofa.value_counts().to_dict()}")

        print(f"Feature selection complete:")
        print(f"  Using {X.shape[1]} embedding features")
        print(f"  Target variable: {self.target_variable} (binary classification)")
        print(f"  Class distribution: {y.value_counts().to_dict()}")
        print(f"  Excluded all clinical features to prevent data leakage")

        # Check for any string values that would cause conversion errors
        if X.dtypes.apply(lambda x: x == 'object').any():
            raise ValueError("String values detected in embedding features. Check data types.")

        if X_mofa is not None and X_mofa.dtypes.apply(lambda x: x == 'object').any():
            raise ValueError("String values detected in MOFA embedding features. Check data types.")

        if self.benchmark and X_mofa is not None:
            return X, y, X_mofa, y_mofa
        else:
            return X, y

    def train_and_evaluate_models(self, test_size=0.2, n_folds=5, random_state=42, run=None, n_trials=30, optimize_hyperparams=True):
        """Train and evaluate multiple models on the tumor purity classification task.

        Uses k-fold cross-validation on the training set and evaluates on a separate test set.
        If benchmark is True, also trains and evaluates models on MOFA embeddings for comparison.

        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
            n_folds (int): Number of folds for cross-validation (default: 5)
            random_state (int): Random seed for reproducibility (default: 42)
            run (wandb.Run, optional): Weights & Biases run for logging
            n_trials (int): Number of Optuna trials for hyperparameter optimization
            optimize_hyperparams (bool): Whether to optimize hyperparameters

        Returns:
            dict: Results for each model
        """
        # Start timing
        start_time = time.time()

        # Get features and target
        if self.benchmark and hasattr(self, 'mofa_merged_df') and self.mofa_merged_df is not None:
            X, y, X_mofa, y_mofa = self.get_model_data()
            print("\n--- Running with MOFA benchmark comparison ---")
        else:
            X, y = self.get_model_data()
            X_mofa, y_mofa = None, None

        # Split data into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        print(f"Primary model - Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Primary model - Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

        # Split MOFA data if available
        X_mofa_train, X_mofa_test, y_mofa_train, y_mofa_test = None, None, None, None
        if X_mofa is not None and y_mofa is not None:
            # Instead of using indices directly, we need to ensure we're using the same patients
            # Get the patient IDs for the train and test sets
            if 'patient_id' in self.merged_df.columns:
                train_patient_ids = self.merged_df.loc[y_train.index, 'patient_id'].values
                test_patient_ids = self.merged_df.loc[y_test.index, 'patient_id'].values

                # Get the indices in the MOFA DataFrame that correspond to these patient IDs
                if 'patient_id' in self.mofa_merged_df.columns:
                    # Create a mapping from patient ID to index in MOFA DataFrame
                    mofa_patient_id_to_idx = {pid: idx for idx, pid in zip(self.mofa_merged_df.index, self.mofa_merged_df['patient_id'])}

                    # Log information about patient overlap
                    primary_patient_ids = set(train_patient_ids) | set(test_patient_ids)
                    mofa_patient_ids = set(self.mofa_merged_df['patient_id'])
                    common_patients = primary_patient_ids & mofa_patient_ids

                    print(f"Patient ID overlap information:")
                    print(f"  Primary dataset: {len(primary_patient_ids)} unique patients")
                    print(f"  MOFA dataset: {len(mofa_patient_ids)} unique patients")
                    print(f"  Common patients: {len(common_patients)} patients")

                    # Get the indices for the train and test sets
                    mofa_train_indices = [mofa_patient_id_to_idx[pid] for pid in train_patient_ids if pid in mofa_patient_id_to_idx]
                    mofa_test_indices = [mofa_patient_id_to_idx[pid] for pid in test_patient_ids if pid in mofa_patient_id_to_idx]

                    # Check if we have enough patients in common
                    if len(mofa_train_indices) == 0 or len(mofa_test_indices) == 0:
                        print("Warning: No common patients found between primary and MOFA datasets for train or test set.")
                        print("Cannot perform benchmark comparison. Please check your data.")
                        return {}

                    # Extract MOFA data for the same patients
                    X_mofa_train = X_mofa.loc[mofa_train_indices]
                    X_mofa_test = X_mofa.loc[mofa_test_indices]
                    y_mofa_train = y_mofa.loc[mofa_train_indices]
                    y_mofa_test = y_mofa.loc[mofa_test_indices]

                    print(f"MOFA benchmark - Training data: {X_mofa_train.shape[0]} samples, {X_mofa_train.shape[1]} features")
                    print(f"MOFA benchmark - Testing data: {X_mofa_test.shape[0]} samples, {X_mofa_test.shape[1]} features")

                    # Log information about the train/test split
                    print(f"Train/test split information:")
                    print(f"  Primary train set: {len(train_patient_ids)} patients")
                    print(f"  Primary test set: {len(test_patient_ids)} patients")
                    print(f"  MOFA train set: {len(mofa_train_indices)} patients")
                    print(f"  MOFA test set: {len(mofa_test_indices)} patients")
                    print(f"  Train set overlap: {len(mofa_train_indices)}/{len(train_patient_ids)} patients ({len(mofa_train_indices)/len(train_patient_ids)*100:.1f}%)")
                    print(f"  Test set overlap: {len(mofa_test_indices)}/{len(test_patient_ids)} patients ({len(mofa_test_indices)/len(test_patient_ids)*100:.1f}%)")
                else:
                    print("Warning: 'patient_id' column not found in MOFA DataFrame. Cannot align MOFA data with primary data.")
            else:
                print("Warning: 'patient_id' column not found in primary DataFrame. Cannot align MOFA data with primary data.")

        # Log to wandb if available
        if run is not None:
            run.log({
                "data/train_samples": X_train.shape[0],
                "data/test_samples": X_test.shape[0],
                "data/features": X_train.shape[1],
                "data/positive_ratio": y.mean(),
                "data/cancer_type": self.cancer_type,
                "data/embedding_type": self.embedding_type,
                "data/benchmark": self.benchmark
            })

            if X_mofa is not None:
                run.log({
                    "data/mofa_features": X_mofa.shape[1],
                    "data/mofa_positive_ratio": y_mofa.mean()
                })

        # Feature scaling for primary embeddings
        # Apply StandardScaler followed by MinMaxScaler for robust scaling
        scaler_std = StandardScaler()
        scaler_minmax = MinMaxScaler()

        X_train_scaled = scaler_std.fit_transform(X_train)
        X_train_scaled = scaler_minmax.fit_transform(X_train_scaled)

        # Transform the test data using the same scalers
        X_test_scaled = scaler_std.transform(X_test)
        X_test_scaled = scaler_minmax.transform(X_test_scaled)

        print("Primary model - Feature scaling applied: StandardScaler followed by MinMaxScaler")

        # Feature scaling for MOFA embeddings if available
        scaler_std_mofa = None
        scaler_minmax_mofa = None
        X_mofa_train_scaled = None
        X_mofa_test_scaled = None

        if X_mofa_train is not None and X_mofa_test is not None:
            scaler_std_mofa = StandardScaler()
            scaler_minmax_mofa = MinMaxScaler()

            X_mofa_train_scaled = scaler_std_mofa.fit_transform(X_mofa_train)
            X_mofa_train_scaled = scaler_minmax_mofa.fit_transform(X_mofa_train_scaled)

            X_mofa_test_scaled = scaler_std_mofa.transform(X_mofa_test)
            X_mofa_test_scaled = scaler_minmax_mofa.transform(X_mofa_test_scaled)

            print("MOFA benchmark - Feature scaling applied: StandardScaler followed by MinMaxScaler")

        # Define models to evaluate
        if optimize_hyperparams:
            # Optimize hyperparameters using Optuna
            print("\nOptimizing hyperparameters using Optuna...")
            best_params = self.optimize_hyperparameters(
                X_train=X_train_scaled,
                y_train=y_train,
                n_folds=n_folds,
                n_trials=n_trials,
                random_state=random_state,
                run=run
            )

            # Define models with optimized hyperparameters
            models = {
                'SVM': SVC(**best_params['SVM'], probability=True, random_state=random_state),
                'KNN': KNeighborsClassifier(**best_params['KNN']),
                'Random Forest': RandomForestClassifier(**best_params['Random Forest'], random_state=random_state),
                'XGBoost': XGBClassifier(**best_params['XGBoost'], random_state=random_state)
            }
        else:
            # Use default models without hyperparameter optimization
            print("\nUsing default model configurations (no hyperparameter optimization)")
            models = {
                'SVM': SVC(probability=True, random_state=random_state),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(random_state=random_state),
                'XGBoost': XGBClassifier(random_state=random_state)
            }

        # Initialize results dictionary
        results = {}

        # Set up k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Create a tqdm progress bar for models
        model_pbar = tqdm(models.items(), desc=f"Models ({self.cancer_type}, {self.embedding_type})",
                          position=0, leave=True)

        # Train and evaluate each model
        for name, model in model_pbar:
            model_pbar.set_description(f"Model: {name} ({self.cancer_type}, {self.embedding_type})")

            # Initialize lists to store cross-validation results
            cv_accuracy = []
            cv_precision = []
            cv_recall = []
            cv_f1 = []
            cv_auc = []

            # Create a tqdm progress bar for folds
            folds = list(skf.split(X_train_scaled, y_train))
            fold_pbar = tqdm(enumerate(folds, 1), desc=f"CV Folds", total=n_folds,
                             position=1, leave=False)

            # Perform k-fold cross-validation on the training set
            for fold_idx, (train_idx, val_idx) in fold_pbar:
                fold_pbar.set_description(f"Fold {fold_idx}/{n_folds}")

                # Split data into training and validation sets for this fold
                X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train the model on this fold
                model.fit(X_train_fold, y_train_fold)

                # Make predictions on the validation set
                y_val_pred = model.predict(X_val_fold)
                y_val_prob = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics for this fold
                fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
                fold_precision = precision_score(y_val_fold, y_val_pred)
                fold_recall = recall_score(y_val_fold, y_val_pred)
                fold_f1 = f1_score(y_val_fold, y_val_pred)
                fold_auc = roc_auc_score(y_val_fold, y_val_prob) if y_val_prob is not None else None

                # Store metrics for this fold
                cv_accuracy.append(fold_accuracy)
                cv_precision.append(fold_precision)
                cv_recall.append(fold_recall)
                cv_f1.append(fold_f1)
                if fold_auc is not None:
                    cv_auc.append(fold_auc)

                # Update progress bar with metrics
                fold_pbar.set_postfix({
                    'Accuracy': f"{fold_accuracy:.4f}",
                    'F1': f"{fold_f1:.4f}"
                })

                # Log fold results to wandb if available
                if run is not None:
                    run.log({
                        f"{self.cancer_type}/{self.embedding_type}/{name}/fold_{fold_idx}/accuracy": fold_accuracy,
                        f"{self.cancer_type}/{self.embedding_type}/{name}/fold_{fold_idx}/precision": fold_precision,
                        f"{self.cancer_type}/{self.embedding_type}/{name}/fold_{fold_idx}/recall": fold_recall,
                        f"{self.cancer_type}/{self.embedding_type}/{name}/fold_{fold_idx}/f1": fold_f1,
                        f"{self.cancer_type}/{self.embedding_type}/{name}/fold_{fold_idx}/auc": fold_auc if fold_auc is not None else 0
                    })

            # Calculate mean and standard deviation of cross-validation metrics
            mean_accuracy = np.mean(cv_accuracy)
            std_accuracy = np.std(cv_accuracy)
            mean_precision = np.mean(cv_precision)
            std_precision = np.std(cv_precision)
            mean_recall = np.mean(cv_recall)
            std_recall = np.std(cv_recall)
            mean_f1 = np.mean(cv_f1)
            std_f1 = np.std(cv_f1)
            mean_auc = np.mean(cv_auc) if cv_auc else None
            std_auc = np.std(cv_auc) if cv_auc else None

            print(f"\n  Cross-validation results for {name}:")
            print(f"    Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"    Precision: {mean_precision:.4f} ± {std_precision:.4f}")
            print(f"    Recall: {mean_recall:.4f} ± {std_recall:.4f}")
            print(f"    F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
            if mean_auc is not None:
                print(f"    AUC: {mean_auc:.4f} ± {std_auc:.4f}")

            # Log cross-validation results to wandb if available
            if run is not None:
                run.log({
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/mean_accuracy": mean_accuracy,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/mean_precision": mean_precision,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/mean_recall": mean_recall,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/mean_f1": mean_f1,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/mean_auc": mean_auc if mean_auc is not None else 0,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/std_accuracy": std_accuracy,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/std_precision": std_precision,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/std_recall": std_recall,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/std_f1": std_f1,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/cv/std_auc": std_auc if std_auc is not None else 0
                })

            # Train the final model on the entire training set
            print(f"\n  Training final {name} model on entire training set...")
            model.fit(X_train_scaled, y_train)

            # Evaluate on the test set
            y_test_pred = model.predict(X_test_scaled)
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics on the test set
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else None

            print(f"\n  Test set results for {name}:")
            print(f"    Accuracy: {test_accuracy:.4f}")
            print(f"    Precision: {test_precision:.4f}")
            print(f"    Recall: {test_recall:.4f}")
            print(f"    F1 Score: {test_f1:.4f}")
            if test_auc is not None:
                print(f"    AUC: {test_auc:.4f}")

            # Log test results to wandb if available
            if run is not None:
                run.log({
                    f"{self.cancer_type}/{self.embedding_type}/{name}/test/accuracy": test_accuracy,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/test/precision": test_precision,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/test/recall": test_recall,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/test/f1": test_f1,
                    f"{self.cancer_type}/{self.embedding_type}/{name}/test/auc": test_auc if test_auc is not None else 0
                })

                # Create and log confusion matrix as a figure
                cm = confusion_matrix(y_test, y_test_pred)
                plt.figure(figsize=(8, 6))

                # Set appropriate labels based on the target variable
                if self.target_variable == 'pathology_T_stage':
                    labels = ['Early (T1/T2)', 'Advanced (T3/T4)']
                elif self.target_variable == 'pathology_N_stage':
                    labels = ['Negative (N0)', 'Positive (N1/N2)']
                elif self.target_variable == 'Tumor_purity':
                    labels = ['Low', 'High']
                else:
                    labels = ['Class 0', 'Class 1']

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix - {name} ({self.cancer_type.upper()}, {self.embedding_type.capitalize()})')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()

                # Log figure to wandb
                run.log({f"{self.cancer_type}/{self.embedding_type}/{name}/confusion_matrix": wandb.Image(plt)})
                plt.close()

            # Store results
            results[name] = {
                # Cross-validation results
                'cv_accuracy': cv_accuracy,
                'cv_precision': cv_precision,
                'cv_recall': cv_recall,
                'cv_f1': cv_f1,
                'cv_auc': cv_auc if cv_auc else None,
                'mean_cv_accuracy': mean_accuracy,
                'mean_cv_precision': mean_precision,
                'mean_cv_recall': mean_recall,
                'mean_cv_f1': mean_f1,
                'mean_cv_auc': mean_auc,
                'std_cv_accuracy': std_accuracy,
                'std_cv_precision': std_precision,
                'std_cv_recall': std_recall,
                'std_cv_f1': std_f1,
                'std_cv_auc': std_auc,

                # Test set results
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,

                # Model and predictions
                'model': model,
                'predictions': y_test_pred,
                'probabilities': y_test_prob,

                # Scalers (for future use)
                'scaler_std': scaler_std,
                'scaler_minmax': scaler_minmax
            }

            # Update model progress bar
            model_pbar.set_postfix({
                'Test Acc': f"{test_accuracy:.4f}",
                'Test F1': f"{test_f1:.4f}"
            })

        # If benchmark is True and MOFA data is available, train and evaluate models on MOFA embeddings
        if self.benchmark and X_mofa_train_scaled is not None and X_mofa_test_scaled is not None:
            print("\n" + "="*80)
            print(f"Training and evaluating models on MOFA embeddings for benchmarking")
            print("="*80)

            # Initialize MOFA results dictionary
            mofa_results = {}

            # Create a tqdm progress bar for models
            mofa_model_pbar = tqdm(models.items(), desc=f"MOFA Models ({self.cancer_type})",
                                  position=0, leave=True)

            # Train and evaluate each model on MOFA embeddings
            for name, model_template in mofa_model_pbar:
                mofa_model_pbar.set_description(f"MOFA Model: {name} ({self.cancer_type})")

                # Create a fresh instance of the model to avoid any influence from previous training
                if isinstance(model_template, SVC):
                    model = SVC(**model_template.get_params())
                elif isinstance(model_template, KNeighborsClassifier):
                    model = KNeighborsClassifier(**model_template.get_params())
                elif isinstance(model_template, RandomForestClassifier):
                    model = RandomForestClassifier(**model_template.get_params())
                elif isinstance(model_template, XGBClassifier):
                    model = XGBClassifier(**model_template.get_params())
                else:
                    # Fallback for any other model type
                    model = type(model_template)(**model_template.get_params())

                # Initialize lists to store cross-validation results
                cv_accuracy = []
                cv_precision = []
                cv_recall = []
                cv_f1 = []
                cv_auc = []

                # Create a tqdm progress bar for folds
                folds = list(skf.split(X_mofa_train_scaled, y_mofa_train))
                fold_pbar = tqdm(enumerate(folds, 1), desc=f"MOFA CV Folds", total=n_folds,
                                position=1, leave=False)

                # Perform k-fold cross-validation on the MOFA training set
                for fold_idx, (train_idx, val_idx) in fold_pbar:
                    fold_pbar.set_description(f"MOFA Fold {fold_idx}/{n_folds}")

                    # Split data into training and validation sets for this fold
                    X_train_fold, X_val_fold = X_mofa_train_scaled[train_idx], X_mofa_train_scaled[val_idx]
                    y_train_fold, y_val_fold = y_mofa_train.iloc[train_idx], y_mofa_train.iloc[val_idx]

                    # Train the model on this fold
                    model.fit(X_train_fold, y_train_fold)

                    # Make predictions on the validation set
                    y_val_pred = model.predict(X_val_fold)
                    y_val_prob = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Calculate metrics for this fold
                    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
                    fold_precision = precision_score(y_val_fold, y_val_pred)
                    fold_recall = recall_score(y_val_fold, y_val_pred)
                    fold_f1 = f1_score(y_val_fold, y_val_pred)
                    fold_auc = roc_auc_score(y_val_fold, y_val_prob) if y_val_prob is not None else None

                    # Store metrics for this fold
                    cv_accuracy.append(fold_accuracy)
                    cv_precision.append(fold_precision)
                    cv_recall.append(fold_recall)
                    cv_f1.append(fold_f1)
                    if fold_auc is not None:
                        cv_auc.append(fold_auc)

                    # Update progress bar with metrics
                    fold_pbar.set_postfix({
                        'Accuracy': f"{fold_accuracy:.4f}",
                        'F1': f"{fold_f1:.4f}"
                    })

                    # Log fold results to wandb if available
                    if run is not None:
                        run.log({
                            f"{self.cancer_type}/mofa/{name}/fold_{fold_idx}/accuracy": fold_accuracy,
                            f"{self.cancer_type}/mofa/{name}/fold_{fold_idx}/precision": fold_precision,
                            f"{self.cancer_type}/mofa/{name}/fold_{fold_idx}/recall": fold_recall,
                            f"{self.cancer_type}/mofa/{name}/fold_{fold_idx}/f1": fold_f1,
                            f"{self.cancer_type}/mofa/{name}/fold_{fold_idx}/auc": fold_auc if fold_auc is not None else 0
                        })

                # Calculate mean and standard deviation of cross-validation metrics
                mean_accuracy = np.mean(cv_accuracy)
                std_accuracy = np.std(cv_accuracy)
                mean_precision = np.mean(cv_precision)
                std_precision = np.std(cv_precision)
                mean_recall = np.mean(cv_recall)
                std_recall = np.std(cv_recall)
                mean_f1 = np.mean(cv_f1)
                std_f1 = np.std(cv_f1)
                mean_auc = np.mean(cv_auc) if cv_auc else None
                std_auc = np.std(cv_auc) if cv_auc else None

                print(f"\n  MOFA Cross-validation results for {name}:")
                print(f"    Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
                print(f"    Precision: {mean_precision:.4f} ± {std_precision:.4f}")
                print(f"    Recall: {mean_recall:.4f} ± {std_recall:.4f}")
                print(f"    F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
                if mean_auc is not None:
                    print(f"    AUC: {mean_auc:.4f} ± {std_auc:.4f}")

                # Log cross-validation results to wandb if available
                if run is not None:
                    run.log({
                        f"{self.cancer_type}/mofa/{name}/cv/mean_accuracy": mean_accuracy,
                        f"{self.cancer_type}/mofa/{name}/cv/mean_precision": mean_precision,
                        f"{self.cancer_type}/mofa/{name}/cv/mean_recall": mean_recall,
                        f"{self.cancer_type}/mofa/{name}/cv/mean_f1": mean_f1,
                        f"{self.cancer_type}/mofa/{name}/cv/mean_auc": mean_auc if mean_auc is not None else 0,
                        f"{self.cancer_type}/mofa/{name}/cv/std_accuracy": std_accuracy,
                        f"{self.cancer_type}/mofa/{name}/cv/std_precision": std_precision,
                        f"{self.cancer_type}/mofa/{name}/cv/std_recall": std_recall,
                        f"{self.cancer_type}/mofa/{name}/cv/std_f1": std_f1,
                        f"{self.cancer_type}/mofa/{name}/cv/std_auc": std_auc if std_auc is not None else 0
                    })

                # Train the final model on the entire MOFA training set
                print(f"\n  Training final MOFA {name} model on entire training set...")
                model.fit(X_mofa_train_scaled, y_mofa_train)

                # Evaluate on the MOFA test set
                y_test_pred = model.predict(X_mofa_test_scaled)
                y_test_prob = model.predict_proba(X_mofa_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics on the test set
                test_accuracy = accuracy_score(y_mofa_test, y_test_pred)
                test_precision = precision_score(y_mofa_test, y_test_pred)
                test_recall = recall_score(y_mofa_test, y_test_pred)
                test_f1 = f1_score(y_mofa_test, y_test_pred)
                test_auc = roc_auc_score(y_mofa_test, y_test_prob) if y_test_prob is not None else None

                print(f"\n  MOFA Test set results for {name}:")
                print(f"    Accuracy: {test_accuracy:.4f}")
                print(f"    Precision: {test_precision:.4f}")
                print(f"    Recall: {test_recall:.4f}")
                print(f"    F1 Score: {test_f1:.4f}")
                if test_auc is not None:
                    print(f"    AUC: {test_auc:.4f}")

                # Log test results to wandb if available
                if run is not None:
                    run.log({
                        f"{self.cancer_type}/mofa/{name}/test/accuracy": test_accuracy,
                        f"{self.cancer_type}/mofa/{name}/test/precision": test_precision,
                        f"{self.cancer_type}/mofa/{name}/test/recall": test_recall,
                        f"{self.cancer_type}/mofa/{name}/test/f1": test_f1,
                        f"{self.cancer_type}/mofa/{name}/test/auc": test_auc if test_auc is not None else 0
                    })

                    # Create and log confusion matrix as a figure
                    cm = confusion_matrix(y_mofa_test, y_test_pred)
                    plt.figure(figsize=(8, 6))

                    # Set appropriate labels based on the target variable
                    if self.target_variable == 'pathology_T_stage':
                        labels = ['Early (T1/T2)', 'Advanced (T3/T4)']
                    elif self.target_variable == 'pathology_N_stage':
                        labels = ['Negative (N0)', 'Positive (N1/N2)']
                    elif self.target_variable == 'Tumor_purity':
                        labels = ['Low', 'High']
                    else:
                        labels = ['Class 0', 'Class 1']

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f'Confusion Matrix - {name} ({self.cancer_type.upper()}, MOFA)')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.tight_layout()

                    # Log figure to wandb
                    run.log({f"{self.cancer_type}/mofa/{name}/confusion_matrix": wandb.Image(plt)})
                    plt.close()

                # Store MOFA results
                mofa_results[name] = {
                    # Cross-validation results
                    'cv_accuracy': cv_accuracy,
                    'cv_precision': cv_precision,
                    'cv_recall': cv_recall,
                    'cv_f1': cv_f1,
                    'cv_auc': cv_auc if cv_auc else None,
                    'mean_cv_accuracy': mean_accuracy,
                    'mean_cv_precision': mean_precision,
                    'mean_cv_recall': mean_recall,
                    'mean_cv_f1': mean_f1,
                    'mean_cv_auc': mean_auc,
                    'std_cv_accuracy': std_accuracy,
                    'std_cv_precision': std_precision,
                    'std_cv_recall': std_recall,
                    'std_cv_f1': std_f1,
                    'std_cv_auc': std_auc,

                    # Test set results
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'test_auc': test_auc,

                    # Model and predictions
                    'model': model,
                    'predictions': y_test_pred,
                    'probabilities': y_test_prob,

                    # Scalers (for future use)
                    'scaler_std': scaler_std_mofa,
                    'scaler_minmax': scaler_minmax_mofa
                }

                # Update model progress bar
                mofa_model_pbar.set_postfix({
                    'Test Acc': f"{test_accuracy:.4f}",
                    'Test F1': f"{test_f1:.4f}"
                })

            # Add MOFA results to the main results dictionary
            results['mofa_results'] = mofa_results

            print("\n" + "="*80)
            print(f"MOFA benchmark evaluation complete")
            print("="*80)

        # Calculate and log total execution time
        execution_time = time.time() - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

        if run is not None:
            run.log({f"{self.cancer_type}/{self.embedding_type}/execution_time": execution_time})

        return results

    def optimize_hyperparameters(self, X_train, y_train, n_folds=5, n_trials=50, random_state=42, run=None):
        """Optimize hyperparameters for each model using Optuna.

        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training labels
            n_folds (int): Number of folds for cross-validation
            n_trials (int): Number of Optuna trials
            random_state (int): Random seed for reproducibility
            run (wandb.Run, optional): Weights & Biases run for logging

        Returns:
            dict: Best hyperparameters for each model
        """
        # Set up k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Initialize dictionary to store best hyperparameters
        best_params = {}

        # Define model types to optimize
        model_types = ['SVM', 'KNN', 'Random Forest', 'XGBoost']

        # Create a tqdm progress bar for models
        model_pbar = tqdm(model_types, desc=f"Optimizing Models ({self.cancer_type}, {self.embedding_type})",
                          position=0, leave=True)

        for model_type in model_pbar:
            model_pbar.set_description(f"Optimizing: {model_type} ({self.cancer_type}, {self.embedding_type})")

            # Create a study for this model type
            study = optuna.create_study(
                sampler=TPESampler(seed=random_state),
                pruner=HyperbandPruner(),
                direction="maximize",  # Maximize F1 score
                study_name=f"{model_type}_{self.cancer_type}_{self.embedding_type}"
            )

            # Define the objective function for this model type
            def objective(trial):
                # Define hyperparameters based on model type
                if model_type == 'SVM':
                    params = {
                        'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                        'probability': True,
                        'random_state': random_state
                    }
                    model = SVC(**params)

                elif model_type == 'KNN':
                    params = {
                        'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                        'p': trial.suggest_int('p', 1, 2)  # 1 for manhattan, 2 for euclidean
                    }
                    model = KNeighborsClassifier(**params)

                elif model_type == 'Random Forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'random_state': random_state
                    }
                    model = RandomForestClassifier(**params)

                elif model_type == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': random_state
                    }
                    model = XGBClassifier(**params)

                # Perform k-fold cross-validation
                f1_scores = []

                # Create a tqdm progress bar for folds (only in the first trial)
                folds = list(skf.split(X_train, y_train))
                fold_iter = enumerate(folds, 1)

                if trial.number == 0:
                    fold_iter = tqdm(fold_iter, desc=f"CV Folds", total=n_folds,
                                    position=1, leave=False)

                for fold_idx, (train_idx, val_idx) in fold_iter:
                    # Split data for this fold
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    # Train the model
                    model.fit(X_train_fold, y_train_fold)

                    # Make predictions
                    y_val_pred = model.predict(X_val_fold)

                    # Calculate F1 score
                    fold_f1 = f1_score(y_val_fold, y_val_pred)
                    f1_scores.append(fold_f1)

                    # Report intermediate result for pruning
                    trial.report(fold_f1, fold_idx - 1)

                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                # Return mean F1 score across all folds
                mean_f1 = np.mean(f1_scores)
                return mean_f1

            # Optimize with progress tracking
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            # Get best parameters
            best_params[model_type] = study.best_params
            best_f1 = study.best_value

            print(f"\n  Best {model_type} parameters: {best_params[model_type]}")
            print(f"  Best cross-validation F1 score: {best_f1:.4f}")

            # Log to wandb if available
            if run is not None:
                # Log best parameters and score
                for param_name, param_value in best_params[model_type].items():
                    run.log({f"{self.cancer_type}/{self.embedding_type}/{model_type}/best_{param_name}": param_value})
                run.log({f"{self.cancer_type}/{self.embedding_type}/{model_type}/best_cv_f1": best_f1})

                # Log optimization plots if possible
                try:
                    # Create and log optimization history plot
                    fig_history = plot_optimization_history(study)
                    run.log({f"{self.cancer_type}/{self.embedding_type}/{model_type}/optimization_history": wandb.Image(fig_history)})
                    plt.close()

                    # Create and log parameter importance plot
                    fig_importance = plot_param_importances(study)
                    run.log({f"{self.cancer_type}/{self.embedding_type}/{model_type}/param_importance": wandb.Image(fig_importance)})
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create Optuna visualization: {e}")

        return best_params

    def visualize_results(self, results, run=None):
        """Visualize model performance and tumor purity distribution.

        If benchmark is True, also creates comparative visualizations with MOFA embeddings.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a subdirectory for this specific analysis
        if self.benchmark:
            analysis_dir = os.path.join(self.output_dir, f"{self.cancer_type}_{self.embedding_type}_benchmark")
        else:
            analysis_dir = os.path.join(self.output_dir, f"{self.cancer_type}_{self.embedding_type}")
        os.makedirs(analysis_dir, exist_ok=True)

        # Plot target variable distribution based on the selected target
        if self.target_variable == 'pathology_T_stage':
            # Plot T stage distribution
            plt.figure(figsize=(10, 6))
            t_stage_counts = self.clinical_df_binary['t_stage_category'].value_counts()
            ax = sns.barplot(x=t_stage_counts.index, y=t_stage_counts.values)
            plt.title(f'T Stage Distribution - {self.cancer_type.upper()} Cancer')
            plt.xlabel('T Stage Category')
            plt.ylabel('Count')

            # Add value labels on top of bars
            for i, v in enumerate(t_stage_counts.values):
                ax.text(i, v + 5, str(v), ha='center')

            plt.savefig(os.path.join(analysis_dir, 't_stage_distribution.png'))

            # Log to wandb if available
            if run is not None:
                run.log({f"{self.cancer_type}/t_stage_distribution": wandb.Image(plt)})

            plt.close()

        elif self.target_variable == 'pathology_N_stage':
            # Plot N stage distribution
            plt.figure(figsize=(10, 6))
            n_stage_counts = self.clinical_df_binary['n_stage_category'].value_counts()
            ax = sns.barplot(x=n_stage_counts.index, y=n_stage_counts.values)
            plt.title(f'N Stage Distribution - {self.cancer_type.upper()} Cancer')
            plt.xlabel('N Stage Category')
            plt.ylabel('Count')

            # Add value labels on top of bars
            for i, v in enumerate(n_stage_counts.values):
                ax.text(i, v + 5, str(v), ha='center')

            plt.savefig(os.path.join(analysis_dir, 'n_stage_distribution.png'))

            # Log to wandb if available
            if run is not None:
                run.log({f"{self.cancer_type}/n_stage_distribution": wandb.Image(plt)})

            plt.close()

        elif self.target_variable == 'Tumor_purity':
            # Plot tumor purity distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.clinical_df['Tumor_purity'], bins=20, kde=True)
            plt.axvline(self.clinical_df['Tumor_purity'].quantile(0.25), color='r', linestyle='--', label='Q1 (Low)')
            plt.axvline(self.clinical_df['Tumor_purity'].quantile(0.75), color='g', linestyle='--', label='Q3 (High)')
            plt.title(f'Tumor Purity Distribution - {self.cancer_type.upper()} Cancer')
            plt.xlabel('Tumor Purity')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(analysis_dir, 'tumor_purity_distribution.png'))

            # Also create a bar plot of the binary categories
            plt.figure(figsize=(10, 6))
            purity_counts = self.clinical_df_binary['tumor_purity_category'].value_counts()
            ax = sns.barplot(x=purity_counts.index, y=purity_counts.values)
            plt.title(f'Tumor Purity Categories - {self.cancer_type.upper()} Cancer')
            plt.xlabel('Tumor Purity Category')
            plt.ylabel('Count')

            # Add value labels on top of bars
            for i, v in enumerate(purity_counts.values):
                ax.text(i, v + 5, str(v), ha='center')

            plt.savefig(os.path.join(analysis_dir, 'tumor_purity_categories.png'))

            # Log to wandb if available
            if run is not None:
                run.log({f"{self.cancer_type}/tumor_purity_distribution": wandb.Image(plt)})

            plt.close()

        # Plot model performance comparison
        if results:
            # Extract cross-validation metrics for comparison
            cv_metrics = ['mean_cv_accuracy', 'mean_cv_precision', 'mean_cv_recall', 'mean_cv_f1', 'mean_cv_auc']
            cv_metrics_std = ['std_cv_accuracy', 'std_cv_precision', 'std_cv_recall', 'std_cv_f1', 'std_cv_auc']
            test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']

            # Simplified metric names for plotting
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

            # Get model names, excluding special keys like 'mofa_results' and 'benchmark_comparison'
            model_names = [name for name in results.keys() if name not in ['mofa_results', 'benchmark_comparison']]

            # Create DataFrames for plotting
            cv_performance_df = pd.DataFrame(index=model_names, columns=metric_labels)
            cv_std_df = pd.DataFrame(index=model_names, columns=metric_labels)
            test_performance_df = pd.DataFrame(index=model_names, columns=metric_labels)

            for model_name in model_names:
                for i, metric in enumerate(cv_metrics):
                    if metric in results[model_name] and results[model_name][metric] is not None:
                        cv_performance_df.loc[model_name, metric_labels[i]] = results[model_name][metric]
                        cv_std_df.loc[model_name, metric_labels[i]] = results[model_name][cv_metrics_std[i]]

                for i, metric in enumerate(test_metrics):
                    if metric in results[model_name] and results[model_name][metric] is not None:
                        test_performance_df.loc[model_name, metric_labels[i]] = results[model_name][metric]

            # Plot cross-validation performance comparison
            plt.figure(figsize=(14, 8))
            ax = cv_performance_df.plot(kind='bar', yerr=cv_std_df, capsize=4, figsize=(14, 8))
            plt.title(f'Cross-Validation Performance - {self.cancer_type.upper()} Cancer ({self.embedding_type.capitalize()} Embeddings)')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.legend(loc='lower right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on top of bars - safely handle different container types
            for i, container in enumerate(ax.containers):
                # Skip error bars which are typically every other container when using yerr
                if i % 2 == 0:  # Only label the bars, not the error bars
                    try:
                        ax.bar_label(container, fmt='%.2f', padding=3)
                    except (AttributeError, TypeError):
                        # If container doesn't support bar_label, we'll add text manually
                        if hasattr(container, 'datavalues'):
                            # For line plots
                            pass
                        elif hasattr(container, 'patches'):
                            # For regular bar plots
                            for rect in container.patches:
                                height = rect.get_height()
                                ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                                        f'{height:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'cv_performance_comparison.png'))

            # Log to wandb if available
            if run is not None:
                run.log({f"{self.cancer_type}/{self.embedding_type}/cv_performance_comparison": wandb.Image(plt)})

            plt.close()

            # Plot test set performance comparison
            plt.figure(figsize=(14, 8))
            ax = test_performance_df.plot(kind='bar', figsize=(14, 8))
            plt.title(f'Test Set Performance - {self.cancer_type.upper()} Cancer ({self.embedding_type.capitalize()} Embeddings)')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.legend(loc='lower right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on top of bars - safely handle different container types
            for i, container in enumerate(ax.containers):
                try:
                    ax.bar_label(container, fmt='%.2f', padding=3)
                except (AttributeError, TypeError):
                    # If container doesn't support bar_label, we'll add text manually
                    if hasattr(container, 'patches'):
                        # For regular bar plots
                        for rect in container.patches:
                            height = rect.get_height()
                            ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                                    f'{height:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'test_performance_comparison.png'))

            # Log to wandb if available
            if run is not None:
                run.log({f"{self.cancer_type}/{self.embedding_type}/test_performance_comparison": wandb.Image(plt)})

            plt.close()

            # Plot confusion matrices
            for name, result in results.items():
                if 'predictions' in result:
                    # Get the actual target values for the test set
                    y_true = self.merged_df[self.target_column].iloc[len(self.merged_df) - len(result['predictions']):]

                    # Create confusion matrix
                    cm = confusion_matrix(y_true, result['predictions'])
                    plt.figure(figsize=(8, 6))

                    # Set appropriate labels based on the target variable
                    if self.target_variable == 'pathology_T_stage':
                        labels = ['Early (T1/T2)', 'Advanced (T3/T4)']
                    elif self.target_variable == 'pathology_N_stage':
                        labels = ['Negative (N0)', 'Positive (N1/N2)']
                    elif self.target_variable == 'Tumor_purity':
                        labels = ['Low', 'High']
                    else:
                        labels = ['Class 0', 'Class 1']

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f'Confusion Matrix - {name}\n{self.cancer_type.upper()} Cancer ({self.embedding_type.capitalize()} Embeddings)')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.tight_layout()
                    plt.savefig(os.path.join(analysis_dir, f'confusion_matrix_{name}.png'))

                    # Log to wandb if available
                    if run is not None:
                        run.log({f"{self.cancer_type}/{self.embedding_type}/confusion_matrix_{name}": wandb.Image(plt)})

                    plt.close()

            # Save detailed results to CSV
            cv_results_df = pd.DataFrame()
            test_results_df = pd.DataFrame()

            # Add metadata to results
            metadata = pd.DataFrame({
                'Cancer Type': [self.cancer_type.upper()],
                'Embedding Type': [self.embedding_type.capitalize()],
                'Analysis Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            })

            for model_name in model_names:
                # Cross-validation results
                for i, metric in enumerate(cv_metrics):
                    if metric in results[model_name] and results[model_name][metric] is not None:
                        cv_results_df.loc[model_name, f'{metric_labels[i]} (Mean)'] = results[model_name][metric]
                        cv_results_df.loc[model_name, f'{metric_labels[i]} (Std)'] = results[model_name][cv_metrics_std[i]]

                # Test set results
                for i, metric in enumerate(test_metrics):
                    if metric in results[model_name] and results[model_name][metric] is not None:
                        test_results_df.loc[model_name, metric_labels[i]] = results[model_name][metric]

            # Save to CSV
            metadata.to_csv(os.path.join(analysis_dir, 'analysis_metadata.csv'), index=False)
            cv_results_df.to_csv(os.path.join(analysis_dir, 'cross_validation_results.csv'))
            test_results_df.to_csv(os.path.join(analysis_dir, 'test_set_results.csv'))

            # Create a boxplot of cross-validation results for each model
            plt.figure(figsize=(15, 10))

            # Prepare data for boxplot
            boxplot_data = []
            boxplot_labels = []

            for i, metric in enumerate(['cv_accuracy', 'cv_f1', 'cv_auc']):
                for name in model_names:
                    if metric in results[name] and results[name][metric] is not None:
                        boxplot_data.append(results[name][metric])
                        boxplot_labels.append(f"{name} - {metric.replace('cv_', '')}")

            if boxplot_data:
                plt.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
                plt.title(f'Cross-Validation Performance Distribution\n{self.cancer_type.upper()} Cancer ({self.embedding_type.capitalize()} Embeddings)')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, 'cv_performance_boxplot.png'))

                # Log to wandb if available
                if run is not None:
                    run.log({f"{self.cancer_type}/{self.embedding_type}/cv_performance_boxplot": wandb.Image(plt)})

                plt.close()

            # Create benchmark comparison visualizations if benchmark is True
            if self.benchmark and 'mofa_results' in results:
                mofa_results = results.pop('mofa_results')  # Extract MOFA results

                # Create a DataFrame for benchmark comparison
                benchmark_df = pd.DataFrame(columns=['Primary', 'MOFA'], index=metric_labels)

                # Find the best model for each embedding type
                best_primary_model = None
                best_primary_f1 = 0
                best_mofa_model = None
                best_mofa_f1 = 0

                # Find best model for primary embeddings
                for model_name in model_names:
                    if 'test_f1' in results[model_name] and results[model_name]['test_f1'] > best_primary_f1:
                        best_primary_f1 = results[model_name]['test_f1']
                        best_primary_model = model_name

                # Find best model for MOFA embeddings
                # Make sure we only compare actual models, not special keys
                mofa_model_names = [name for name in mofa_results.keys()
                                   if name not in ['mofa_results', 'benchmark_comparison']]
                for model_name in mofa_model_names:
                    if 'test_f1' in mofa_results[model_name] and mofa_results[model_name]['test_f1'] > best_mofa_f1:
                        best_mofa_f1 = mofa_results[model_name]['test_f1']
                        best_mofa_model = model_name

                # Create comparison of best models
                print(f"\nBenchmark Comparison - Best Models:")
                print(f"  Primary ({self.embedding_type}): {best_primary_model} (F1: {best_primary_f1:.4f})")
                print(f"  MOFA: {best_mofa_model} (F1: {best_mofa_f1:.4f})")

                # Fill benchmark DataFrame with best model metrics
                for i, metric in enumerate(test_metrics):
                    if metric in results[best_primary_model]:
                        benchmark_df.loc[metric_labels[i], 'Primary'] = results[best_primary_model][metric]
                    if metric in mofa_results[best_mofa_model]:
                        benchmark_df.loc[metric_labels[i], 'MOFA'] = mofa_results[best_mofa_model][metric]

                # Plot benchmark comparison
                plt.figure(figsize=(12, 8))
                ax = benchmark_df.plot(kind='bar', figsize=(12, 8))
                plt.title(f'Benchmark Comparison: {self.embedding_type.capitalize()} vs MOFA\n{self.cancer_type.upper()} Cancer - {self.target_variable}')
                plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.legend(title='Embedding Type')
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # Add value labels on top of bars
                for i, container in enumerate(ax.containers):
                    try:
                        ax.bar_label(container, fmt='%.2f', padding=3)
                    except (AttributeError, TypeError):
                        if hasattr(container, 'patches'):
                            for rect in container.patches:
                                height = rect.get_height()
                                ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                                        f'{height:.2f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, 'benchmark_comparison.png'))

                # Log to wandb if available
                if run is not None:
                    run.log({f"{self.cancer_type}/{self.embedding_type}/benchmark_comparison": wandb.Image(plt)})

                plt.close()

                # Save benchmark comparison to CSV
                benchmark_df.to_csv(os.path.join(analysis_dir, 'benchmark_comparison.csv'))

                # Add benchmark results back to the main results for saving
                results['mofa_results'] = mofa_results
                results['benchmark_comparison'] = {
                    'best_primary_model': best_primary_model,
                    'best_mofa_model': best_mofa_model,
                    'benchmark_metrics': benchmark_df.to_dict()
                }

            # Save the complete results object for future reference
            if self.benchmark:
                results_filename = f'cancer_stage_classification_results_{self.target_variable}_{self.cancer_type}_{self.embedding_type}_benchmark.joblib'
            else:
                results_filename = f'cancer_stage_classification_results_{self.target_variable}_{self.cancer_type}_{self.embedding_type}.joblib'
            joblib.dump(results, os.path.join(analysis_dir, results_filename))

            # Create a summary text file
            with open(os.path.join(analysis_dir, 'analysis_summary.txt'), 'w') as f:
                if self.benchmark:
                    f.write(f"Cancer Stage Classification Analysis Summary (with MOFA Benchmark)\n")
                    f.write(f"=================================================================\n\n")
                else:
                    f.write(f"Cancer Stage Classification Analysis Summary\n")
                    f.write(f"========================================\n\n")

                f.write(f"Target Variable: {self.target_variable}\n")
                f.write(f"Cancer Type: {self.cancer_type.upper()}\n")
                f.write(f"Embedding Type: {self.embedding_type.capitalize()}\n")
                if self.benchmark:
                    f.write(f"Benchmark: MOFA embeddings\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write(f"Dataset Information:\n")
                f.write(f"  - Original clinical data: {len(self.clinical_df)} patients\n")
                f.write(f"  - Binary classification data: {len(self.clinical_df_binary)} patients\n")
                f.write(f"  - Final merged data: {len(self.merged_df)} patients\n\n")

                # Write appropriate target variable information
                if self.target_variable == 'pathology_T_stage':
                    early_count = (self.clinical_df_binary['t_stage_category'] == 'early').sum()
                    advanced_count = (self.clinical_df_binary['t_stage_category'] == 'advanced').sum()

                    f.write(f"T Stage Distribution:\n")
                    f.write(f"  - Early (T1/T2): {early_count} patients\n")
                    f.write(f"  - Advanced (T3/T4): {advanced_count} patients\n\n")
                elif self.target_variable == 'pathology_N_stage':
                    negative_count = (self.clinical_df_binary['n_stage_category'] == 'negative').sum()
                    positive_count = (self.clinical_df_binary['n_stage_category'] == 'positive').sum()

                    f.write(f"N Stage Distribution:\n")
                    f.write(f"  - Negative (N0): {negative_count} patients\n")
                    f.write(f"  - Positive (N1/N2/N3): {positive_count} patients\n\n")
                elif self.target_variable == 'Tumor_purity':
                    low_count = (self.clinical_df_binary['tumor_purity_category'] == 'low').sum()
                    high_count = (self.clinical_df_binary['tumor_purity_category'] == 'high').sum()
                    q1 = self.clinical_df['Tumor_purity'].quantile(0.25)
                    q3 = self.clinical_df['Tumor_purity'].quantile(0.75)

                    f.write(f"Tumor Purity Distribution:\n")
                    f.write(f"  - Low (≤ {q1:.4f}): {low_count} patients\n")
                    f.write(f"  - High (≥ {q3:.4f}): {high_count} patients\n")
                    f.write(f"  - Total for binary classification: {len(self.clinical_df_binary)} patients\n\n")

                f.write(f"Model Performance Summary (Test Set):\n")
                for model_name in model_names:
                    if model_name != 'mofa_results' and model_name != 'benchmark_comparison':
                        f.write(f"  {model_name}:\n")
                        for i, metric in enumerate(test_metrics):
                            if metric in results[model_name] and results[model_name][metric] is not None:
                                f.write(f"    - {metric_labels[i]}: {results[model_name][metric]:.4f}\n")
                        f.write("\n")

                # Add benchmark comparison information if available
                if self.benchmark and 'benchmark_comparison' in results:
                    f.write(f"\nBenchmark Comparison Summary:\n")
                    f.write(f"  Primary ({self.embedding_type}) - Best Model: {results['benchmark_comparison']['best_primary_model']}\n")
                    f.write(f"  MOFA - Best Model: {results['benchmark_comparison']['best_mofa_model']}\n\n")

                    # Add metric comparisons
                    benchmark_metrics = results['benchmark_comparison']['benchmark_metrics']
                    for metric in metric_labels:
                        if metric in benchmark_metrics:
                            primary_value = benchmark_metrics[metric]['Primary']
                            mofa_value = benchmark_metrics[metric]['MOFA']
                            f.write(f"  {metric}:\n")
                            f.write(f"    - Primary ({self.embedding_type}): {primary_value:.4f}\n")
                            f.write(f"    - MOFA: {mofa_value:.4f}\n")

                            # Calculate improvement percentage
                            if mofa_value > 0:
                                improvement = ((primary_value - mofa_value) / mofa_value) * 100
                                if improvement > 0:
                                    f.write(f"    - Improvement: +{improvement:.2f}%\n")
                                else:
                                    f.write(f"    - Difference: {improvement:.2f}%\n")
                            f.write("\n")

def run_single_analysis(cancer_type, embedding_type, data_root, embeddings_root, output_dir, test_size, n_folds, random_state, target_variable='pathology_T_stage', use_wandb=True, n_trials=30, optimize_hyperparams=True, benchmark=False, mofa_embeddings_dir='results/mofa'):
    """Run analysis for a single cancer type and embedding type combination."""
    print(f"\n{'='*50}")
    print(f"Running analysis for {cancer_type.upper()} cancer with {embedding_type.capitalize()} embeddings")
    if benchmark:
        print(f"With MOFA embeddings as benchmark")
    print(f"{'='*50}")

    # Initialize wandb if requested
    run = None
    if use_wandb:
        try:
            # Create a unique run name
            run_name = f"cancer_stage_{target_variable}_{cancer_type}_{embedding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if benchmark:
                run_name += "_benchmark"

            # Initialize wandb run
            run = wandb.init(
                project="cancer-stage-classification",
                name=run_name,
                config={
                    "cancer_type": cancer_type,
                    "embedding_type": embedding_type,
                    "test_size": test_size,
                    "n_folds": n_folds,
                    "random_state": random_state,
                    "optimize_hyperparams": optimize_hyperparams,
                    "n_trials": n_trials if optimize_hyperparams else 0,
                    "benchmark": benchmark,
                    "mofa_embeddings_dir": mofa_embeddings_dir if benchmark else None
                },
                group=f"{cancer_type}_analysis",
                job_type=embedding_type,
                reinit=True
            )

            print(f"Initialized wandb run: {run_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            run = None

    # Create evaluator
    evaluator = CancerStageEvaluator(
        cancer_type=cancer_type,
        embedding_type=embedding_type,
        data_root=data_root,
        embeddings_root=embeddings_root,
        output_dir=output_dir,
        target_variable=target_variable,  # Use the target variable passed to the function
        benchmark=benchmark,
        mofa_embeddings_dir=mofa_embeddings_dir
    )

    # Train and evaluate models
    results = evaluator.train_and_evaluate_models(
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state,
        run=run,
        n_trials=n_trials,
        optimize_hyperparams=optimize_hyperparams
    )

    # Visualize results
    evaluator.visualize_results(results, run=run)

    # Finish wandb run if it was initialized
    if run is not None:
        # Create analysis directory path for artifact logging
        if benchmark:
            analysis_dir = os.path.join(output_dir, f"{cancer_type}_{embedding_type}_benchmark")
            artifact_name = f"cancer_stage_results_{target_variable}_{cancer_type}_{embedding_type}_benchmark"
            artifact_desc = f"Cancer stage classification results for {target_variable} in {cancer_type} cancer using {embedding_type} embeddings with MOFA benchmark"
        else:
            analysis_dir = os.path.join(output_dir, f"{cancer_type}_{embedding_type}")
            artifact_name = f"cancer_stage_results_{target_variable}_{cancer_type}_{embedding_type}"
            artifact_desc = f"Cancer stage classification results for {target_variable} in {cancer_type} cancer using {embedding_type} embeddings"

        # Log result files as artifacts
        if os.path.exists(analysis_dir):
            artifact = wandb.Artifact(
                name=artifact_name,
                type="results",
                description=artifact_desc
            )

            # Add all files in the analysis directory
            for root, _, files in os.walk(analysis_dir):
                for file in files:
                    artifact.add_file(os.path.join(root, file))

            # Log the artifact
            run.log_artifact(artifact)

        # Finish the run
        run.finish()

    # Return results for potential cross-comparison
    return results

def compare_embedding_types(cancer_type, data_root, embeddings_root, output_dir, test_size, n_folds, random_state, target_variable='pathology_T_stage', use_wandb=True, n_trials=30, optimize_hyperparams=True, benchmark=False, mofa_embeddings_dir='results/mofa'):
    """Compare performance across different embedding types for a specific cancer type."""
    embedding_types = ['autoencoder', 'gcn', 'integrated']
    all_results = {}

    # Run analysis for each embedding type
    for embedding_type in embedding_types:
        all_results[embedding_type] = run_single_analysis(
            cancer_type=cancer_type,
            embedding_type=embedding_type,
            data_root=data_root,
            embeddings_root=embeddings_root,
            output_dir=output_dir,
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            target_variable=target_variable,
            use_wandb=use_wandb,
            n_trials=n_trials,
            optimize_hyperparams=optimize_hyperparams,
            benchmark=benchmark,
            mofa_embeddings_dir=mofa_embeddings_dir
        )

    # Create a comparison directory
    comparison_dir = os.path.join(output_dir, f"{cancer_type}_embedding_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Compare performance across embedding types
    print(f"\n{'='*50}")
    print(f"Comparing embedding types for {cancer_type.upper()} cancer")
    print(f"{'='*50}")

    # Initialize wandb for comparison if requested
    comparison_run = None
    if use_wandb:
        try:
            # Create a unique run name for the comparison
            run_name = f"embedding_comparison_{target_variable}_{cancer_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Initialize wandb run
            comparison_run = wandb.init(
                project="cancer-stage-classification",
                name=run_name,
                config={
                    "cancer_type": cancer_type,
                    "embedding_types": embedding_types,
                    "test_size": test_size,
                    "n_folds": n_folds,
                    "random_state": random_state,
                    "comparison_type": "embedding_comparison"
                },
                group=f"{cancer_type}_comparisons",
                job_type="embedding_comparison",
                reinit=True
            )

            print(f"Initialized wandb run for embedding comparison: {run_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb for comparison: {e}")
            comparison_run = None

    # Extract test metrics for comparison
    test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    # For each model type, create a separate comparison
    model_names = list(all_results['autoencoder'].keys())  # Assuming all embedding types have the same models

    for model_name in model_names:
        model_comparison_df = pd.DataFrame(columns=embedding_types, index=metric_labels)

        for embedding_type in embedding_types:
            for i, metric in enumerate(test_metrics):
                if metric in all_results[embedding_type][model_name]:
                    model_comparison_df.loc[metric_labels[i], embedding_type] = all_results[embedding_type][model_name][metric]

        # Plot comparison
        plt.figure(figsize=(12, 8))
        ax = model_comparison_df.plot(kind='bar', figsize=(12, 8))
        plt.title(f'Embedding Type Comparison - {model_name} ({cancer_type.upper()} Cancer)')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Embedding Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars - safely handle different container types
        for i, container in enumerate(ax.containers):
            try:
                ax.bar_label(container, fmt='%.2f', padding=3)
            except (AttributeError, TypeError):
                # If container doesn't support bar_label, we'll add text manually
                if hasattr(container, 'patches'):
                    # For regular bar plots
                    for rect in container.patches:
                        height = rect.get_height()
                        ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                                f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'embedding_comparison_{model_name}.png'))

        # Log to wandb if available
        if comparison_run is not None:
            comparison_run.log({f"embedding_comparison/{cancer_type}/{model_name}": wandb.Image(plt)})

        plt.close()

        # Save comparison to CSV
        model_comparison_df.to_csv(os.path.join(comparison_dir, f'embedding_comparison_{model_name}.csv'))

    # Create an overall comparison across all models
    # For each embedding type and model, get the best metric
    best_metrics = {}
    for embedding_type in embedding_types:
        best_metrics[embedding_type] = {}
        for metric in metric_labels:
            best_value = 0
            best_model = ''
            for model_name in model_names:
                metric_idx = metric_labels.index(metric)
                if test_metrics[metric_idx] in all_results[embedding_type][model_name]:
                    value = all_results[embedding_type][model_name][test_metrics[metric_idx]]
                    if value > best_value:
                        best_value = value
                        best_model = model_name
            best_metrics[embedding_type][metric] = (best_value, best_model)

    # Create a DataFrame for the best metrics
    best_df = pd.DataFrame(columns=embedding_types, index=metric_labels)
    best_models_df = pd.DataFrame(columns=embedding_types, index=metric_labels)

    for embedding_type in embedding_types:
        for metric in metric_labels:
            best_df.loc[metric, embedding_type] = best_metrics[embedding_type][metric][0]
            best_models_df.loc[metric, embedding_type] = best_metrics[embedding_type][metric][1]

    # Plot best metrics comparison
    plt.figure(figsize=(12, 8))
    ax = best_df.plot(kind='bar', figsize=(12, 8))
    plt.title(f'Best Performance Across Models - {cancer_type.upper()} Cancer')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Embedding Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars - safely handle different container types
    for i, container in enumerate(ax.containers):
        try:
            ax.bar_label(container, fmt='%.2f', padding=3)
        except (AttributeError, TypeError):
            # If container doesn't support bar_label, we'll add text manually
            if hasattr(container, 'patches'):
                # For regular bar plots
                for rect in container.patches:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                            f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'best_performance_comparison.png'))

    # Log to wandb if available
    if comparison_run is not None:
        comparison_run.log({f"embedding_comparison/{cancer_type}/best_performance": wandb.Image(plt)})

    plt.close()

    # Save best metrics to CSV
    best_df.to_csv(os.path.join(comparison_dir, 'best_performance_metrics.csv'))
    best_models_df.to_csv(os.path.join(comparison_dir, 'best_performance_models.csv'))

    # Create a summary text file
    with open(os.path.join(comparison_dir, 'embedding_comparison_summary.txt'), 'w') as f:
        f.write(f"Embedding Type Comparison for {cancer_type.upper()} Cancer\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"Best Performance Metrics:\n")
        for metric in metric_labels:
            f.write(f"  {metric}:\n")
            for embedding_type in embedding_types:
                value, model = best_metrics[embedding_type][metric]
                f.write(f"    {embedding_type.capitalize()}: {value:.4f} ({model})\n")
            f.write("\n")

    # Log comparison results as artifacts if wandb is available
    if comparison_run is not None:
        # Create an artifact
        artifact = wandb.Artifact(
            name=f"embedding_comparison_results_{cancer_type}",
            type="comparison",
            description=f"Embedding type comparison results for {cancer_type} cancer"
        )

        # Add all files in the comparison directory
        for root, _, files in os.walk(comparison_dir):
            for file in files:
                artifact.add_file(os.path.join(root, file))

        # Log the artifact
        comparison_run.log_artifact(artifact)

        # Log summary metrics
        for metric in metric_labels:
            for embedding_type in embedding_types:
                value, model = best_metrics[embedding_type][metric]
                comparison_run.log({f"best_{metric.lower()}_{embedding_type}": value})

        # Finish the run
        comparison_run.finish()

    print(f"Embedding type comparison complete. Results saved to {comparison_dir}")
    return all_results

def compare_cancer_types(embedding_type, data_root, embeddings_root, output_dir, test_size, n_folds, random_state, target_variable='pathology_T_stage', use_wandb=True, n_trials=30, optimize_hyperparams=True, benchmark=False, mofa_embeddings_dir='results/mofa'):
    """Compare performance across different cancer types for a specific embedding type."""
    cancer_types = ['colorec', 'panc']
    all_results = {}

    # Run analysis for each cancer type
    for cancer_type in cancer_types:
        all_results[cancer_type] = run_single_analysis(
            cancer_type=cancer_type,
            embedding_type=embedding_type,
            data_root=data_root,
            embeddings_root=embeddings_root,
            output_dir=output_dir,
            test_size=test_size,
            n_folds=n_folds,
            random_state=random_state,
            target_variable=target_variable,
            use_wandb=use_wandb,
            n_trials=n_trials,
            optimize_hyperparams=optimize_hyperparams,
            benchmark=benchmark,
            mofa_embeddings_dir=mofa_embeddings_dir
        )

    # Create a comparison directory
    comparison_dir = os.path.join(output_dir, f"{embedding_type}_cancer_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Compare performance across cancer types
    print(f"\n{'='*50}")
    print(f"Comparing cancer types for {embedding_type.capitalize()} embeddings")
    print(f"{'='*50}")

    # Initialize wandb for comparison if requested
    comparison_run = None
    if use_wandb:
        try:
            # Create a unique run name for the comparison
            run_name = f"cancer_comparison_{embedding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Initialize wandb run
            comparison_run = wandb.init(
                project="tumor-purity-classification",
                name=run_name,
                config={
                    "cancer_types": cancer_types,
                    "embedding_type": embedding_type,
                    "test_size": test_size,
                    "n_folds": n_folds,
                    "random_state": random_state,
                    "comparison_type": "cancer_comparison"
                },
                group=f"{embedding_type}_comparisons",
                job_type="cancer_comparison",
                reinit=True
            )

            print(f"Initialized wandb run for cancer comparison: {run_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb for comparison: {e}")
            comparison_run = None

    # Extract test metrics for comparison
    test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    # For each model type, create a separate comparison
    model_names = list(all_results['colorec'].keys())  # Assuming all cancer types have the same models

    for model_name in model_names:
        model_comparison_df = pd.DataFrame(columns=cancer_types, index=metric_labels)

        for cancer_type in cancer_types:
            for i, metric in enumerate(test_metrics):
                if metric in all_results[cancer_type][model_name]:
                    model_comparison_df.loc[metric_labels[i], cancer_type] = all_results[cancer_type][model_name][metric]

        # Plot comparison
        plt.figure(figsize=(12, 8))
        ax = model_comparison_df.plot(kind='bar', figsize=(12, 8))
        plt.title(f'Cancer Type Comparison - {model_name} ({embedding_type.capitalize()} Embeddings)')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Cancer Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars - safely handle different container types
        for i, container in enumerate(ax.containers):
            try:
                ax.bar_label(container, fmt='%.2f', padding=3)
            except (AttributeError, TypeError):
                # If container doesn't support bar_label, we'll add text manually
                if hasattr(container, 'patches'):
                    # For regular bar plots
                    for rect in container.patches:
                        height = rect.get_height()
                        ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                                f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'cancer_comparison_{model_name}.png'))

        # Log to wandb if available
        if comparison_run is not None:
            comparison_run.log({f"cancer_comparison/{embedding_type}/{model_name}": wandb.Image(plt)})

        plt.close()

        # Save comparison to CSV
        model_comparison_df.to_csv(os.path.join(comparison_dir, f'cancer_comparison_{model_name}.csv'))

    # Create an overall comparison across all models
    # For each cancer type and model, get the best metric
    best_metrics = {}
    for cancer_type in cancer_types:
        best_metrics[cancer_type] = {}
        for metric in metric_labels:
            best_value = 0
            best_model = ''
            for model_name in model_names:
                metric_idx = metric_labels.index(metric)
                if test_metrics[metric_idx] in all_results[cancer_type][model_name]:
                    value = all_results[cancer_type][model_name][test_metrics[metric_idx]]
                    if value > best_value:
                        best_value = value
                        best_model = model_name
            best_metrics[cancer_type][metric] = (best_value, best_model)

    # Create a DataFrame for the best metrics
    best_df = pd.DataFrame(columns=cancer_types, index=metric_labels)
    best_models_df = pd.DataFrame(columns=cancer_types, index=metric_labels)

    for cancer_type in cancer_types:
        for metric in metric_labels:
            best_df.loc[metric, cancer_type] = best_metrics[cancer_type][metric][0]
            best_models_df.loc[metric, cancer_type] = best_metrics[cancer_type][metric][1]

    # Plot best metrics comparison
    plt.figure(figsize=(12, 8))
    ax = best_df.plot(kind='bar', figsize=(12, 8))
    plt.title(f'Best Performance Across Models - {embedding_type.capitalize()} Embeddings')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Cancer Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars - safely handle different container types
    for i, container in enumerate(ax.containers):
        try:
            ax.bar_label(container, fmt='%.2f', padding=3)
        except (AttributeError, TypeError):
            # If container doesn't support bar_label, we'll add text manually
            if hasattr(container, 'patches'):
                # For regular bar plots
                for rect in container.patches:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.03,
                            f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'best_performance_comparison.png'))

    # Log to wandb if available
    if comparison_run is not None:
        comparison_run.log({f"cancer_comparison/{embedding_type}/best_performance": wandb.Image(plt)})

    plt.close()

    # Save best metrics to CSV
    best_df.to_csv(os.path.join(comparison_dir, 'best_performance_metrics.csv'))
    best_models_df.to_csv(os.path.join(comparison_dir, 'best_performance_models.csv'))

    # Create a summary text file
    with open(os.path.join(comparison_dir, 'cancer_comparison_summary.txt'), 'w') as f:
        f.write(f"Cancer Type Comparison for {embedding_type.capitalize()} Embeddings\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"Best Performance Metrics:\n")
        for metric in metric_labels:
            f.write(f"  {metric}:\n")
            for cancer_type in cancer_types:
                value, model = best_metrics[cancer_type][metric]
                f.write(f"    {cancer_type.upper()}: {value:.4f} ({model})\n")
            f.write("\n")

    # Log comparison results as artifacts if wandb is available
    if comparison_run is not None:
        # Create an artifact
        artifact = wandb.Artifact(
            name=f"cancer_comparison_results_{embedding_type}",
            type="comparison",
            description=f"Cancer type comparison results for {embedding_type} embeddings"
        )

        # Add all files in the comparison directory
        for root, _, files in os.walk(comparison_dir):
            for file in files:
                artifact.add_file(os.path.join(root, file))

        # Log the artifact
        comparison_run.log_artifact(artifact)

        # Log summary metrics
        for metric in metric_labels:
            for cancer_type in cancer_types:
                value, model = best_metrics[cancer_type][metric]
                comparison_run.log({f"best_{metric.lower()}_{cancer_type}": value})

        # Finish the run
        comparison_run.finish()

    print(f"Cancer type comparison complete. Results saved to {comparison_dir}")
    return all_results

def compare_all(data_root, embeddings_root, output_dir, test_size, n_folds, random_state, target_variable='pathology_T_stage', use_wandb=True, n_trials=30, optimize_hyperparams=True, benchmark=False, mofa_embeddings_dir='results/mofa'):
    """Compare performance across all combinations of cancer types and embedding types."""
    cancer_types = ['colorec', 'panc']
    embedding_types = ['autoencoder', 'gcn', 'integrated']
    all_results = {}

    # Run analysis for each combination
    for cancer_type in tqdm(cancer_types, desc="Cancer Types", position=0):
        all_results[cancer_type] = {}
        for embedding_type in tqdm(embedding_types, desc=f"Embedding Types for {cancer_type}", position=1, leave=False):
            all_results[cancer_type][embedding_type] = run_single_analysis(
                cancer_type=cancer_type,
                embedding_type=embedding_type,
                data_root=data_root,
                embeddings_root=embeddings_root,
                output_dir=output_dir,
                test_size=test_size,
                n_folds=n_folds,
                random_state=random_state,
                target_variable=target_variable,
                use_wandb=use_wandb,
                n_trials=n_trials,
                optimize_hyperparams=optimize_hyperparams,
                benchmark=benchmark,
                mofa_embeddings_dir=mofa_embeddings_dir
            )

    # Create a comparison directory
    comparison_dir = os.path.join(output_dir, "comprehensive_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Compare performance across all combinations
    print(f"\n{'='*50}")
    print(f"Comprehensive comparison across all cancer types and embedding types")
    print(f"{'='*50}")

    # Initialize wandb for comprehensive comparison if requested
    comprehensive_run = None
    if use_wandb:
        try:
            # Create a unique run name for the comparison
            run_name = f"comprehensive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Initialize wandb run
            comprehensive_run = wandb.init(
                project="tumor-purity-classification",
                name=run_name,
                config={
                    "cancer_types": cancer_types,
                    "embedding_types": embedding_types,
                    "test_size": test_size,
                    "n_folds": n_folds,
                    "random_state": random_state,
                    "comparison_type": "comprehensive_comparison"
                },
                group="comprehensive_comparisons",
                job_type="comprehensive_comparison",
                reinit=True
            )

            print(f"Initialized wandb run for comprehensive comparison: {run_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb for comprehensive comparison: {e}")
            comprehensive_run = None

    # Extract test metrics for comparison
    test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    # For each model type, create a comprehensive comparison
    model_names = list(all_results['colorec']['autoencoder'].keys())  # Assuming all combinations have the same models

    for model_name in model_names:
        # Create a multi-index DataFrame for the comprehensive comparison
        index = pd.MultiIndex.from_product([cancer_types, embedding_types], names=['Cancer Type', 'Embedding Type'])
        comprehensive_df = pd.DataFrame(index=index, columns=metric_labels)

        for cancer_type in cancer_types:
            for embedding_type in embedding_types:
                for i, metric in enumerate(test_metrics):
                    if metric in all_results[cancer_type][embedding_type][model_name]:
                        comprehensive_df.loc[(cancer_type, embedding_type), metric_labels[i]] = \
                            all_results[cancer_type][embedding_type][model_name][metric]

        # Save comprehensive comparison to CSV
        comprehensive_df.to_csv(os.path.join(comparison_dir, f'comprehensive_comparison_{model_name}.csv'))

        # Create a heatmap for each metric
        for i, metric in enumerate(metric_labels):
            # Reshape the data for the heatmap
            heatmap_data = comprehensive_df[metric].unstack(level='Embedding Type')

            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='viridis', vmin=0, vmax=1)
            plt.title(f'{metric} - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'heatmap_{model_name}_{metric}.png'))

            # Log to wandb if available
            if comprehensive_run is not None:
                comprehensive_run.log({f"comprehensive_comparison/heatmap_{model_name}_{metric}": wandb.Image(plt)})

            plt.close()

    # Find the best overall combination for each metric
    best_combinations = {}
    for metric in metric_labels:
        best_value = 0
        best_combo = ('', '', '')
        for cancer_type in cancer_types:
            for embedding_type in embedding_types:
                for model_name in model_names:
                    metric_idx = metric_labels.index(metric)
                    if test_metrics[metric_idx] in all_results[cancer_type][embedding_type][model_name]:
                        value = all_results[cancer_type][embedding_type][model_name][test_metrics[metric_idx]]
                        if value > best_value:
                            best_value = value
                            best_combo = (cancer_type, embedding_type, model_name)
        best_combinations[metric] = (best_value, best_combo)

    # Create a summary text file
    with open(os.path.join(comparison_dir, 'comprehensive_comparison_summary.txt'), 'w') as f:
        f.write(f"Comprehensive Comparison Summary\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"Best Overall Combinations:\n")
        for metric in metric_labels:
            value, (cancer_type, embedding_type, model_name) = best_combinations[metric]
            f.write(f"  {metric}: {value:.4f} - {cancer_type.upper()} cancer, {embedding_type.capitalize()} embeddings, {model_name} model\n")

    # Log comparison results as artifacts if wandb is available
    if comprehensive_run is not None:
        # Create an artifact
        artifact = wandb.Artifact(
            name=f"comprehensive_comparison_results",
            type="comparison",
            description=f"Comprehensive comparison results across all cancer types and embedding types"
        )

        # Add all files in the comparison directory
        for root, _, files in os.walk(comparison_dir):
            for file in files:
                artifact.add_file(os.path.join(root, file))

        # Log the artifact
        comprehensive_run.log_artifact(artifact)

        # Log summary metrics
        for metric in metric_labels:
            value, (cancer_type, embedding_type, model_name) = best_combinations[metric]
            comprehensive_run.log({
                f"best_{metric.lower()}_value": value,
                f"best_{metric.lower()}_cancer_type": cancer_type,
                f"best_{metric.lower()}_embedding_type": embedding_type,
                f"best_{metric.lower()}_model": model_name
            })

        # Create a summary table
        data = []
        for metric in metric_labels:
            value, (cancer_type, embedding_type, model_name) = best_combinations[metric]
            data.append([metric, value, cancer_type, embedding_type, model_name])

        table = wandb.Table(columns=["Metric", "Value", "Cancer Type", "Embedding Type", "Model"], data=data)
        comprehensive_run.log({"best_combinations": table})

        # Finish the run
        comprehensive_run.finish()

    print(f"Comprehensive comparison complete. Results saved to {comparison_dir}")
    return all_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate cancer stage classification using embeddings')
    parser.add_argument('--cancer_type', type=str, required=True, choices=['colorec', 'panc', 'both'],
                        help='Cancer type to analyze (colorec, panc, or both)')
    parser.add_argument('--embedding_type', type=str, required=True, choices=['autoencoder', 'gcn', 'integrated', 'all'],
                        help='Type of embeddings to use (autoencoder, gcn, integrated, or all)')
    parser.add_argument('--target_variable', type=str, default='pathology_T_stage',
                        choices=['pathology_T_stage', 'pathology_N_stage', 'Tumor_purity'],
                        help='Target variable for classification (default: pathology_T_stage). Note: Tumor_purity is only valid for colorectal cancer.')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing cancer-specific data folders')
    parser.add_argument('--embeddings_root', type=str, required=True,
                        help='Root directory containing embedding-specific folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for tracking results and visualizations')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials for Optuna hyperparameter optimization (default: 30)')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                        help='Use Optuna to optimize hyperparameters')
    parser.add_argument('--benchmark', action='store_true',
                        help='Use MOFA embeddings as a benchmark for comparison')
    parser.add_argument('--mofa_embeddings_dir', type=str, default='results/mofa',
                        help='Directory containing MOFA embeddings for benchmarking')

    args = parser.parse_args()

    # Print analysis configuration
    print(f"\n{'='*50}")
    print(f"Cancer Classification Analysis")
    print(f"{'='*50}")

    # Display appropriate target variable description
    if args.target_variable == 'pathology_T_stage':
        target_desc = "Pathologic T Stage (Early vs Advanced)"
    elif args.target_variable == 'pathology_N_stage':
        target_desc = "Pathologic N Stage (Lymph Node Involvement)"
    elif args.target_variable == 'Tumor_purity':
        # Check if cancer type is colorectal
        if args.cancer_type != 'colorec' and args.cancer_type != 'both':
            print(f"WARNING: Tumor_purity target variable is only valid for colorectal cancer (colorec), not for {args.cancer_type}")
        target_desc = "Tumor Purity (Low vs High)"
    else:
        target_desc = args.target_variable

    print(f"Target Variable: {target_desc}")
    print(f"Cancer Type: {args.cancer_type.upper()}")
    print(f"Embedding Type: {args.embedding_type.upper() if args.embedding_type == 'all' else args.embedding_type.capitalize()}")
    print(f"Data Root: {args.data_root}")
    print(f"Embeddings Root: {args.embeddings_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Test Size: {args.test_size}")
    print(f"Number of Folds: {args.n_folds}")
    print(f"Random State: {args.random_state}")
    print(f"Use Weights & Biases: {args.use_wandb}")
    print(f"Optimize Hyperparameters: {args.optimize_hyperparams}")
    if args.optimize_hyperparams:
        print(f"Number of Optuna Trials: {args.n_trials}")
    print(f"{'='*50}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate target variable and cancer type combination
    if args.target_variable == 'Tumor_purity' and args.cancer_type == 'panc':
        print(f"ERROR: Tumor_purity target variable is only valid for colorectal cancer (colorec), not for pancreatic cancer (panc).")
        print(f"Please use --cancer_type colorec when using --target_variable Tumor_purity")
        return

    # Determine which analysis to run based on the arguments
    if args.cancer_type == 'both' and args.embedding_type == 'all':
        # Run comprehensive comparison across all combinations
        compare_all(
            data_root=args.data_root,
            embeddings_root=args.embeddings_root,
            output_dir=args.output_dir,
            test_size=args.test_size,
            n_folds=args.n_folds,
            random_state=args.random_state,
            target_variable=args.target_variable,
            use_wandb=args.use_wandb,
            n_trials=args.n_trials,
            optimize_hyperparams=args.optimize_hyperparams,
            benchmark=args.benchmark,
            mofa_embeddings_dir=args.mofa_embeddings_dir
        )
    elif args.cancer_type == 'both':
        # Compare cancer types for a specific embedding type
        compare_cancer_types(
            embedding_type=args.embedding_type,
            data_root=args.data_root,
            embeddings_root=args.embeddings_root,
            output_dir=args.output_dir,
            test_size=args.test_size,
            n_folds=args.n_folds,
            random_state=args.random_state,
            target_variable=args.target_variable,
            use_wandb=args.use_wandb,
            n_trials=args.n_trials,
            optimize_hyperparams=args.optimize_hyperparams,
            benchmark=args.benchmark,
            mofa_embeddings_dir=args.mofa_embeddings_dir
        )
    elif args.embedding_type == 'all':
        # Compare embedding types for a specific cancer type
        compare_embedding_types(
            cancer_type=args.cancer_type,
            data_root=args.data_root,
            embeddings_root=args.embeddings_root,
            output_dir=args.output_dir,
            test_size=args.test_size,
            n_folds=args.n_folds,
            random_state=args.random_state,
            target_variable=args.target_variable,
            use_wandb=args.use_wandb,
            n_trials=args.n_trials,
            optimize_hyperparams=args.optimize_hyperparams,
            benchmark=args.benchmark,
            mofa_embeddings_dir=args.mofa_embeddings_dir
        )
    else:
        # Run a single analysis
        run_single_analysis(
            cancer_type=args.cancer_type,
            embedding_type=args.embedding_type,
            data_root=args.data_root,
            embeddings_root=args.embeddings_root,
            output_dir=args.output_dir,
            test_size=args.test_size,
            n_folds=args.n_folds,
            random_state=args.random_state,
            target_variable=args.target_variable,
            use_wandb=args.use_wandb,
            n_trials=args.n_trials,
            optimize_hyperparams=args.optimize_hyperparams,
            benchmark=args.benchmark,
            mofa_embeddings_dir=args.mofa_embeddings_dir
        )

    print(f"\nAll analyses complete. Results saved to {args.output_dir}")
    print(f"\nTo run another analysis with different parameters, use:")
    print(f"python evaluate_tumor_purity.py --cancer_type [colorec|panc|both] --embedding_type [autoencoder|gcn|integrated|all] --target_variable [pathology_T_stage|pathology_N_stage|Tumor_purity] --data_root [path] --embeddings_root [path] --output_dir [path]")
    print(f"\nTarget variable options:")
    print(f"  - pathology_T_stage: Primary tumor size/extent (default)")
    print(f"  - pathology_N_stage: Regional lymph node involvement")
    print(f"  - Tumor_purity: Tumor purity (low vs high) - only valid for colorectal cancer")

if __name__ == '__main__':
    main()

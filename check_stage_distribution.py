import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from omics_analyzer import OmicsAnalyzer

# Initialize the analyzer
data_dir = "data"
analyzer = OmicsAnalyzer(data_dir)

# Function to check stage distribution
def check_stage_distribution(cancer_type, data_type='rnaseq'):
    """
    Check the distribution of patients across pathologic stages in the harmonized data.
    
    Args:
        cancer_type: "colorectal"/"colorec" or "pancreatic"/"panc"
        data_type: Type of omics data to check
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {cancer_type} cancer {data_type} data")
    print(f"{'='*50}")
    
    # Load appropriate data
    if cancer_type in ["colorectal", "colorec"]:
        data = analyzer.harmonized_colorec[data_type]
        clinical = analyzer.harmonized_colorec["clinical"]
    elif cancer_type in ["pancreatic", "panc"]:
        data = analyzer.harmonized_panc[data_type]
        clinical = analyzer.harmonized_panc["clinical"]
    else:
        raise ValueError("Invalid cancer_type")
    
    # Check if pathologic_stage exists in clinical data
    if 'pathologic_stage' not in clinical.columns:
        print("WARNING: 'pathologic_stage' column not found in clinical data!")
        print(f"Available clinical columns: {clinical.columns.tolist()}")
        
        # Check for similar column names
        stage_columns = [col for col in clinical.columns if 'stage' in col.lower()]
        if stage_columns:
            print(f"Found possible stage-related columns: {stage_columns}")
            
            # Use the first found stage column
            stage_column = stage_columns[0]
            print(f"Using '{stage_column}' instead of 'pathologic_stage'")
        else:
            print("No stage-related columns found. Cannot proceed with analysis.")
            return
    else:
        stage_column = 'pathologic_stage'
    
    # Show total patient count
    print(f"\nTotal patients in {data_type} data: {len(data)}")
    print(f"Total patients in clinical data: {len(clinical)}")
    
    # Count patients per stage
    stage_counts = clinical[stage_column].value_counts().sort_index()
    print(f"\nPatient counts per stage:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} patients")
    
    # Merge data with clinical info
    merged = data.merge(clinical[['patient_id', stage_column]], on='patient_id')
    print(f"\nPatients after merging {data_type} with clinical data: {len(merged)}")
    
    # Count patients per stage after merging
    merged_stage_counts = merged[stage_column].value_counts().sort_index()
    print(f"\nPatient counts per stage after merging:")
    for stage, count in merged_stage_counts.items():
        print(f"  {stage}: {count} patients")
    
    # Check for potential issues
    if any(count < 3 for count in merged_stage_counts.values):
        print("\nWARNING: Some stages have fewer than 3 patients, which may cause visualization issues with violin plots!")
    
    # Select top variable genes for analysis
    values = merged.iloc[:, 1:-1]  # Exclude patient_id and stage
    n_genes = 5  # Same as in the original function
    top_genes = values.var().sort_values(ascending=False).head(n_genes).index.tolist()
    
    print(f"\nTop {n_genes} most variable genes:")
    for i, gene in enumerate(top_genes):
        print(f"  {i+1}. {gene} (variance: {values[gene].var():.6f})")
    
    # Analyze data distribution for top genes
    print(f"\nData distribution for top genes:")
    for gene in top_genes:
        gene_data = merged[[gene, stage_column]]
        print(f"\n  {gene}:")
        
        # Overall statistics
        print(f"    Range: {gene_data[gene].min():.6f} to {gene_data[gene].max():.6f}")
        print(f"    Mean: {gene_data[gene].mean():.6f}, Median: {gene_data[gene].median():.6f}")
        print(f"    Standard deviation: {gene_data[gene].std():.6f}")
        
        # Statistics by stage
        for stage in merged_stage_counts.index:
            stage_data = gene_data[gene_data[stage_column] == stage]
            if len(stage_data) > 0:
                print(f"    {stage} (n={len(stage_data)}): "
                      f"mean={stage_data[gene].mean():.6f}, "
                      f"range=[{stage_data[gene].min():.6f}, {stage_data[gene].max():.6f}]")
    
    # Plot distribution of values for a better visualization
    plt.figure(figsize=(12, 8))
    for i, gene in enumerate(top_genes):
        plt.subplot(n_genes, 1, i+1)
        for stage in merged_stage_counts.index:
            stage_data = merged[merged[stage_column] == stage][gene]
            if len(stage_data) > 0:
                plt.scatter(
                    [stage] * len(stage_data), 
                    stage_data, 
                    alpha=0.5, 
                    label=f"{stage} (n={len(stage_data)})"
                )
        plt.title(f"{gene} Distribution by Stage")
        plt.ylabel("Value")
        if i == n_genes-1:  # Only add x-label to the bottom plot
            plt.xlabel("Pathologic Stage")
    
    plt.tight_layout()
    plt.savefig(f"{cancer_type}_{data_type}_distribution.png")
    print(f"\nDistribution plot saved as {cancer_type}_{data_type}_distribution.png")

# Run the analysis for colorectal cancer RNA-seq data
check_stage_distribution('colorec', 'rnaseq')

# Optional: Also check methylation and scnv data
# check_stage_distribution('colorec', 'methylation')
# check_stage_distribution('colorec', 'scnv') 
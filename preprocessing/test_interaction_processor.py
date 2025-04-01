from preprocessing.data_extractor import DataExtractor

def main():
    print("Testing interaction data processing...")
    
    # Create an instance of the DataExtractor
    extractor = DataExtractor('./data')
    
    # Process both cancer types
    results = extractor.process_interaction_data(cancer_type="both", identifier_type="Official Symbol")
    
    # Display basic information about the processed data
    if 'colorec' in results and results['colorec']:
        colorec_data = results['colorec']
        if isinstance(colorec_data, dict) and 'adjacency_matrix' in colorec_data:
            # If we got the saved paths
            print("\ncolorec cancer interaction data saved to:")
            for key, path in colorec_data.items():
                print(f"  - {key}: {path}")
        else:
            # If we got the actual processed data
            print("\ncolorec cancer interaction data summary:")
            print(f"  - Adjacency matrix shape: {colorec_data['adjacency_matrix'].shape}")
            print(f"  - Number of nodes: {len(colorec_data['node_list'])}")
            print(f"  - Number of core genes: {len(colorec_data['core_gene_indices'])}")
            
            # Print some example gene mappings
            print("\nExample gene mappings (first 5):")
            gene_mapping = colorec_data['gene_mapping']
            for i, (gene, idx) in enumerate(list(gene_mapping.items())[:5]):
                print(f"  - {gene}: {idx}")
    else:
        print("No data processed for colorec cancer.")
        
    if 'panc' in results and results['panc']:
        panc_data = results['panc']
        if isinstance(panc_data, dict) and 'adjacency_matrix' in panc_data:
            # If we got the saved paths
            print("\npanc cancer interaction data saved to:")
            for key, path in panc_data.items():
                print(f"  - {key}: {path}")
        else:
            # If we got the actual processed data
            print("\npanc cancer interaction data summary:")
            print(f"  - Adjacency matrix shape: {panc_data['adjacency_matrix'].shape}")
            print(f"  - Number of nodes: {len(panc_data['node_list'])}")
            print(f"  - Number of core genes: {len(panc_data['core_gene_indices'])}")
            
            # Print some example gene mappings
            print("\nExample gene mappings (first 5):")
            gene_mapping = panc_data['gene_mapping']
            for i, (gene, idx) in enumerate(list(gene_mapping.items())[:5]):
                print(f"  - {gene}: {idx}")
    else:
        print("No data processed for panc cancer.")
    
    print("\nInteraction processing test complete!")

if __name__ == "__main__":
    main() 
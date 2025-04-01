from data_extractor import DataExtractor

# Create an instance of the DataExtractor class
extractor = DataExtractor('./data')

# Run the gene coverage check
results = extractor.check_gene_coverage_in_interactions()

print("\nSummary of coverage results:")
print(f"colorec cancer - Total coverage: {results['colorec_coverage_percent']:.2f}%")
print(f"  - Found in systematic names: {results['colorec_found_in_systematic']}")
print(f"  - Found in official symbols: {results['colorec_found_in_official']}")
print(f"  - Found in synonyms: {results['colorec_found_in_synonyms']}")

print(f"panc cancer - Total coverage: {results['panc_coverage_percent']:.2f}%")
print(f"  - Found in systematic names: {results['panc_found_in_systematic']}")
print(f"  - Found in official symbols: {results['panc_found_in_official']}")
print(f"  - Found in synonyms: {results['panc_found_in_synonyms']}")

# Check if there are any missing genes
if results['colorec_missing_genes'] > 0:
    print(f"\nNumber of missing genes in colorec cancer: {results['colorec_missing_genes']}")
    print(f"Examples: {results['colorec_missing_genes_list'][:5]}")

if results['panc_missing_genes'] > 0:
    print(f"\nNumber of missing genes in panc cancer: {results['panc_missing_genes']}")
    print(f"Examples: {results['panc_missing_genes_list'][:5]}")

print("\nDone!") 
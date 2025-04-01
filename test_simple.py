from data_extractor import DataExtractor

# Create an instance of the DataExtractor class
extractor = DataExtractor('./data')

# Run the gene coverage check
extractor.check_gene_coverage_in_interactions() 
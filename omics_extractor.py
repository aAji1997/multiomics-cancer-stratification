import os
import csv

class OmicsExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def convert_to_csv(self):
        """
        Converts a TSI file to CSV format and saves it in the same directory as the source file.
        
        The TSI file format is tab-separated with attributes as rows and samples as columns.
        The first column contains attribute names, and the first row contains sample IDs.
        The data is transposed during conversion, so rows become columns and vice versa.
        
        Returns:
            str: Path to the generated CSV file
        """

        
        # Get the directory and filename without extension
        directory = os.path.dirname(self.file_path)
        filename = os.path.basename(self.file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Create output path with csv extension
        output_path = os.path.join(directory, f"{filename_without_ext}.csv")
        
        try:
            # Read the entire TSI file into memory
            with open(self.file_path, 'r') as tsi_file:
                # Read all lines and split by tabs
                rows = [line.strip().split('\t') for line in tsi_file]
            
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
                
            print(f"Successfully converted and transposed {self.file_path} to {output_path}")
            return output_path
                
        except Exception as e:
            print(f"Error converting file: {e}")
            return None
        
if __name__ == "__main__":
    extractor = OmicsExtractor("data/clinical.tsi")
    extractor.convert_to_csv()

    

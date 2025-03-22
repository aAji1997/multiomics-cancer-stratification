import os
import csv

class OmicsExtractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

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
        
if __name__ == "__main__":
    extractor = OmicsExtractor("./data")
    extractor.convert_to_csv()

    

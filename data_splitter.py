"""
Script to split the large prepared_data_both.joblib file into smaller cancer-specific files
"""
import os
import joblib
import sys

def split_large_joblib():
    """
    Split the large prepared_data_both.joblib file into smaller cancer-specific files
    """
    source_file = 'data/prepared_data_both.joblib'
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found.")
        return False
    
    try:
        print(f"Loading data from {source_file}...")
        data = joblib.load(source_file)
        
        # Split by cancer type
        print("Splitting data by cancer type...")
        for cancer_type in data.keys():
            output_file = f'data/prepared_data_{cancer_type}.joblib'
            print(f"Creating {output_file}...")
            joblib.dump(data[cancer_type], output_file)
            print(f"Saved {output_file} successfully!")
        
        print("\nDo you want to remove the original large file? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            os.remove(source_file)
            print(f"Removed original file {source_file}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting to split large joblib file...")
    success = split_large_joblib()
    
    if success:
        print("\nProcess completed successfully!")
        print("You can now commit the smaller files with Git LFS.")
    else:
        print("\nProcess failed. Please check the errors above.") 
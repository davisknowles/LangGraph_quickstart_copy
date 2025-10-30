from dotenv import load_dotenv
import os
import pandas as pd
import requests
from io import StringIO

load_dotenv()

def test_blob_connection():
    """Test blob storage connection and list available files."""
    
    blob_sas_url = os.getenv('blob_SAS_url')
    print(f"SAS URL: {blob_sas_url[:50]}...")  # Show first 50 chars for security
    
    # Method 1: Try direct HTTP request to list container contents
    try:
        # Modify URL for listing blobs
        list_url = blob_sas_url.replace('?', '?restype=container&comp=list&')
        print(f"Listing URL: {list_url[:100]}...")
        
        response = requests.get(list_url)
        print(f"List response status: {response.status_code}")
        
        if response.status_code == 200:
            print("Container contents:")
            print(response.text[:500])  # First 500 chars
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error with HTTP method: {str(e)}")
    
    # Method 2: Try to access a specific known file
    try:
        # Let's try to directly access any CSV file
        file_patterns = [
            'nearmiss_data_top100_20251003.csv',
            'safety_data.csv',
            'sample_safety_data.csv'
        ]
        
        base_url = blob_sas_url.split('?')[0]
        sas_params = blob_sas_url.split('?')[1] if '?' in blob_sas_url else ''
        
        for filename in file_patterns:
            file_url = f"{base_url}/{filename}?{sas_params}"
            print(f"\nTrying file: {filename}")
            
            response = requests.head(file_url)  # Use HEAD to check if file exists
            print(f"File check status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"Found file: {filename}")
                # Try to download it
                response = requests.get(file_url)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    print(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
                    print(f"Columns: {list(df.columns[:5])}")
                    return df
                    
    except Exception as e:
        print(f"Error with direct file access: {str(e)}")
    
    return None

if __name__ == "__main__":
    print("Testing blob storage connection...")
    result = test_blob_connection()
    if result is not None:
        print("Success! Blob connection working.")
    else:
        print("Failed to connect to blob storage.")
#!/usr/bin/env python3
"""
Transform and clean hazards & near misses CSV file for Azure Blob Storage and Azure AI Search.

This script:
1. Reads the raw CSV file
2. Cleans and normalizes column headers
3. Transforms data types appropriately
4. Removes BOM and encoding issues
5. Standardizes field formats for search indexing
6. Outputs a cleaned CSV file ready for blob storage
"""

import pandas as pd
import re
from datetime import datetime
import numpy as np
from pathlib import Path

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize column names for Azure AI Search compatibility.
    
    Azure AI Search field naming requirements:
    - Must start with letter or underscore
    - Can contain letters, digits, underscores, hyphens
    - Cannot contain spaces or special characters
    - Case insensitive but preserving original case
    """
    df_clean = df.copy()
    
    # Create mapping of old to new column names
    column_mapping = {}
    
    for col in df.columns:
        # Remove BOM if present
        clean_col = col.replace('ï»¿', '').strip()
        
        # Replace spaces and special characters with underscores
        clean_col = re.sub(r'[^\w\s-]', '', clean_col)  # Remove special chars except spaces and hyphens
        clean_col = re.sub(r'[\s/]+', '_', clean_col)    # Replace spaces and slashes with underscores
        clean_col = re.sub(r'-+', '_', clean_col)        # Replace hyphens with underscores
        clean_col = re.sub(r'_+', '_', clean_col)        # Replace multiple underscores with single
        clean_col = clean_col.strip('_')                 # Remove leading/trailing underscores
        
        # Ensure it starts with letter or underscore
        if clean_col and not (clean_col[0].isalpha() or clean_col[0] == '_'):
            clean_col = f"field_{clean_col}"
        
        # Handle empty column names
        if not clean_col:
            clean_col = f"column_{df.columns.get_loc(col)}"
        
        column_mapping[col] = clean_col
    
    # Rename columns
    df_clean = df_clean.rename(columns=column_mapping)
    
    print("Column name transformations:")
    for old, new in column_mapping.items():
        if old != new:
            print(f"  '{old}' -> '{new}'")
    
    return df_clean, column_mapping

def clean_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize data types for better search indexing.
    """
    df_clean = df.copy()
    
    # Date columns - convert to proper datetime format
    date_columns = ['Created', 'Date_Identified']
    for col in date_columns:
        if col in df_clean.columns:
            try:
                # Handle various date formats
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                print(f"Converted {col} to datetime format")
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
    
    # Boolean columns
    boolean_columns = ['Resolved', 'Issue_Resolved']
    for col in boolean_columns:
        if col in df_clean.columns:
            try:
                # Convert TRUE/FALSE strings to boolean
                df_clean[col] = df_clean[col].map({
                    'TRUE': True, 'True': True, 'true': True, True: True,
                    'FALSE': False, 'False': False, 'false': False, False: False,
                    'Issue Resolved': True, 'Not Resolved': False,
                    '': None, 'NULL': None, np.nan: None
                })
                print(f"Converted {col} to boolean format")
            except Exception as e:
                print(f"Warning: Could not convert {col} to boolean: {e}")
    
    # Numeric columns
    if 'Row_ID' in df_clean.columns:
        try:
            # Extract numeric part from Row ID (e.g., "25-1165" -> 1165)
            df_clean['Row_Number'] = df_clean['Row_ID'].str.extract(r'(\d+)$').astype(float)
            print("Extracted numeric Row_Number from Row_ID")
        except Exception as e:
            print(f"Warning: Could not extract numeric Row_Number: {e}")
    
    # Clean text fields - remove extra whitespace and handle nulls
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        if col not in date_columns and col not in boolean_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(['NULL', 'null', 'nan', '', 'None'], np.nan)
    
    return df_clean

def add_search_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add metadata fields useful for Azure AI Search.
    """
    df_enhanced = df.copy()
    
    # Add a unique document ID for search indexing
    df_enhanced['document_id'] = df_enhanced.index.astype(str)
    
    # Add full-text search field combining key text fields
    text_fields = ['Description_of_Safety_Concern', 'Safety_Comments', 'Specific_Conditions', 
                   'Safety_Concern_Resolution', 'Other_Explain']
    
    search_text_parts = []
    for field in text_fields:
        if field in df_enhanced.columns:
            # Convert to string and handle nulls
            field_text = df_enhanced[field].fillna('').astype(str)
            search_text_parts.append(field_text)
    
    # Combine all text fields for full-text search
    if search_text_parts:
        # Concatenate Series objects properly
        df_enhanced['searchable_content'] = search_text_parts[0]
        for part in search_text_parts[1:]:
            df_enhanced['searchable_content'] = df_enhanced['searchable_content'] + ' | ' + part
        df_enhanced['searchable_content'] = df_enhanced['searchable_content'].str.strip(' |')
        print("Added searchable_content field for full-text search")
    
    # Add processing timestamp
    df_enhanced['processed_timestamp'] = datetime.now().isoformat()
    
    # Create category tags for easier filtering
    if 'Type_of_Near_Miss' in df_enhanced.columns:
        df_enhanced['incident_category'] = df_enhanced['Type_of_Near_Miss'].fillna('Unspecified')
    
    if 'Status' in df_enhanced.columns:
        df_enhanced['is_resolved'] = df_enhanced['Status'].isin(['Complete', 'Resolved', 'Closed'])
    
    return df_enhanced

def validate_for_search(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate the dataframe for Azure AI Search compatibility.
    """
    issues = []
    
    # Check for required fields
    required_fields = ['document_id']
    for field in required_fields:
        if field not in df.columns:
            issues.append(f"Missing required field: {field}")
    
    # Check field name compliance
    for col in df.columns:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', col):
            issues.append(f"Invalid field name for search: '{col}'")
    
    # Check for overly long text fields (Azure AI Search has limits)
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        max_length = df[col].astype(str).str.len().max()
        if max_length > 8000:  # Azure AI Search limit for searchable fields
            issues.append(f"Field '{col}' has text longer than 8000 characters (max: {max_length})")
    
    return len(issues) == 0, issues

def transform_hazard_file(input_file: str, output_file: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Main transformation function.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
    
    Returns:
        Tuple of (transformed_dataframe, transformation_log)
    """
    print(f"Starting transformation of {input_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {e}")
    
    transformation_log = {
        'original_columns': list(df.columns),
        'original_shape': df.shape,
        'start_time': datetime.now().isoformat()
    }
    
    # Step 1: Clean column names
    df_clean, column_mapping = clean_column_names(df)
    transformation_log['column_mapping'] = column_mapping
    
    # Step 2: Clean data types
    df_clean = clean_data_types(df_clean)
    
    # Step 3: Add search metadata
    df_clean = add_search_metadata(df_clean)
    
    # Step 4: Validate for search compatibility
    is_valid, issues = validate_for_search(df_clean)
    transformation_log['validation_issues'] = issues
    transformation_log['is_search_ready'] = is_valid
    
    if not is_valid:
        print("Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Data is ready for Azure AI Search")
    
    transformation_log['final_shape'] = df_clean.shape
    transformation_log['final_columns'] = list(df_clean.columns)
    transformation_log['end_time'] = datetime.now().isoformat()
    
    # Save to output file if specified
    if output_file:
        try:
            df_clean.to_csv(output_file, index=False)
            print(f"Transformed data saved to {output_file}")
            transformation_log['output_file'] = output_file
        except Exception as e:
            print(f"Warning: Failed to save output file: {e}")
    
    print(f"Transformation complete. Final shape: {df_clean.shape}")
    return df_clean, transformation_log

def main():
    """
    Main execution function.
    """
    # File paths
    input_file = Path(__file__).parent.parent / "hazards_near_misses_2025.csv"
    output_file = Path(__file__).parent.parent / "hazards_near_misses_2025_cleaned.csv"
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        df_transformed, log = transform_hazard_file(str(input_file), str(output_file))
        
        # Print summary
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY")
        print("="*60)
        print(f"Original shape: {log['original_shape']}")
        print(f"Final shape: {log['final_shape']}")
        print(f"Columns added: {len(log['final_columns']) - len(log['original_columns'])}")
        
        if log['validation_issues']:
            print(f"Validation issues: {len(log['validation_issues'])}")
        else:
            print("✓ Ready for Azure AI Search and Blob Storage")
        
        # Show sample of transformed data
        print("\nSample of transformed data:")
        print(df_transformed.head(2).to_string())
        
        print(f"\nCleaned file saved as: {output_file}")
        
    except Exception as e:
        print(f"Transformation failed: {e}")

if __name__ == "__main__":
    main()

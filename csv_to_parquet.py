#!/usr/bin/env python3
"""
CSV to Parquet Converter for You-Tune

This script recursively converts all CSV files in a directory to Parquet format,
which significantly improves read performance and reduces storage requirements.
"""

import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("csv2parquet")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert CSV files to Parquet format recursively")
    parser.add_argument(
        "--input-dir", "-i", 
        required=True, 
        help="Input directory containing CSV files (searched recursively)"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        help="Output directory for Parquet files (defaults to same as input)"
    )
    parser.add_argument(
        "--delimiter", "-d", 
        default=",", 
        help="CSV delimiter (default: comma)"
    )
    parser.add_argument(
        "--chunk-size", "-c", 
        type=int, 
        default=100000, 
        help="Chunk size for processing large files (default: 100,000 rows)"
    )
    parser.add_argument(
        "--compression", 
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        default="snappy", 
        help="Compression algorithm for Parquet files (default: snappy)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing Parquet files"
    )
    parser.add_argument(
        "--csv-extension", 
        default=".csv", 
        help="File extension for CSV files (default: .csv)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def find_csv_files(directory, extension):
    """Find all CSV files in directory and subdirectories."""
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension.lower()):
                csv_files.append(os.path.join(root, file))
    return csv_files

def convert_mixed_types(df):
    """
    Handle mixed data types in DataFrame columns by converting appropriately.
    Focuses especially on columns that might contain numeric values as strings.
    """
    # Copy the dataframe to avoid modifying the original
    df_fixed = df.copy()
    
    # List of columns that might need conversion
    numeric_columns = ['like_count', 'dislike_count', 'reply_count', 'view_count', 'subscriber_count']
    timestamp_columns = ['published_at_unix', 'timestamp', 'unix_timestamp', 'created_at_unix']
    boolean_columns = ['is_reply', 'is_spam', 'is_deleted', 'is_popular', 'is_approved', 'is_enabled']
    
    for col in df.columns:
        # Handle boolean columns
        if col in boolean_columns or any(bool_name in col.lower() for bool_name in ['is_', 'has_', '_flag', '_enabled']):
            if df[col].dtype == 'object':  # If column is string/object type
                try:
                    # Convert to lowercase strings first
                    df_fixed[col] = df_fixed[col].astype(str).str.lower()
                    
                    # Map common boolean string representations to actual booleans
                    bool_map = {
                        'true': True, 'yes': True, '1': True, 't': True, 'y': True,
                        'false': False, 'no': False, '0': False, 'f': False, 'n': False
                    }
                    
                    # Apply the mapping
                    df_fixed[col] = df_fixed[col].map(bool_map)
                    
                    # Handle any values that didn't map (keep as strings)
                    unmapped = df_fixed[col].isna() & ~df[col].isna()
                    if unmapped.any():
                        logger.debug(f"Some values in column '{col}' could not be mapped to boolean")
                except Exception as e:
                    logger.debug(f"Could not convert column '{col}' to boolean: {str(e)}")
                    # Keep original values
                    df_fixed[col] = df[col]
        
        # Handle timestamp columns
        elif col in timestamp_columns or any(ts_name in col.lower() for ts_name in ['_unix', 'timestamp', 'epoch']):
            if df[col].dtype == 'object':  # If column is string/object type
                try:
                    # Convert all values to strings first to ensure consistent handling
                    df_fixed[col] = df_fixed[col].astype(str)
                    
                    # Try to detect if values contain decimal points
                    sample = df_fixed[col].dropna().head(100).tolist()
                    has_decimal = any('.' in str(val) for val in sample if str(val) != 'nan')
                    
                    # Attempt numeric conversion
                    if has_decimal:
                        # Keep as float if decimals are present
                        df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                        df_fixed[col] = df_fixed[col].fillna(0)
                    else:
                        # Try integer first, then string if that fails
                        try:
                            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                            df_fixed[col] = df_fixed[col].fillna(0)
                        except:
                            # Keep as string if conversion fails
                            df_fixed[col] = df_fixed[col].astype(str)
                except Exception as e:
                    logger.debug(f"Could not convert timestamp column '{col}': {str(e)}")
                    # Keep as string if all else fails
                    df_fixed[col] = df_fixed[col].astype(str)
        
        # For known potentially numeric columns or any column with 'count' or 'id' in the name
        elif col in numeric_columns or 'count' in col.lower() or '_id' in col.lower():
            if df[col].dtype == 'object':  # If column is string/object type
                try:
                    # Try converting to numeric, coercing errors to NaN
                    df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                    # Replace NaN with 0 for numeric columns
                    df_fixed[col] = df_fixed[col].fillna(0).astype('Int64')
                except Exception as e:
                    logger.debug(f"Could not convert column '{col}' to numeric: {str(e)}")
    
    return df_fixed

def convert_csv_to_parquet(
    csv_path, 
    parquet_path, 
    delimiter=",", 
    chunk_size=100000, 
    compression="snappy", 
    overwrite=False
):
    """Convert a single CSV file to Parquet format."""
    # Check if output file already exists
    if os.path.exists(parquet_path) and not overwrite:
        logger.info(f"Skipping {csv_path} (output file already exists)")
        return False
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    try:
        # Get the total line count for progress tracking
        total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8', errors='replace'))
        rows_processed = 0
        
        # Process the file in chunks
        with tqdm(total=total_rows, desc=f"Converting {os.path.basename(csv_path)}") as pbar:
            # Use pandas to read all data if file is small enough
            if total_rows <= chunk_size:
                df = pd.read_csv(
                    csv_path,
                    sep=delimiter,
                    encoding='utf-8',
                    on_bad_lines='warn',
                    engine='c',
                    low_memory=False  # Prevents mixed type warnings
                )
                
                # Handle mixed data types
                df = convert_mixed_types(df)
                
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    parquet_path,
                    compression=None if compression == "none" else compression
                )
                pbar.update(len(df))
                rows_processed = len(df)
            else:
                # For larger files, process in chunks
                all_data = []
                for chunk in pd.read_csv(
                    csv_path, 
                    sep=delimiter, 
                    chunksize=chunk_size, 
                    low_memory=False,  # Prevents mixed type warnings
                    encoding='utf-8', 
                    on_bad_lines='warn',
                    engine='c'
                ):
                    # Handle mixed data types in each chunk
                    chunk = convert_mixed_types(chunk)
                    all_data.append(chunk)
                    rows_processed += len(chunk)
                    pbar.update(len(chunk))
                
                # Combine all chunks and write at once
                df = pd.concat(all_data, ignore_index=True)
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    parquet_path,
                    compression=None if compression == "none" else compression
                )
        
        # Verify the conversion
        try:
            # Read the file to verify it exists and is valid
            test_read = pq.read_table(parquet_path)
            # Just check that we can access at least one row
            if len(test_read) > 0:
                logger.info(f"Successfully converted {csv_path} → {parquet_path} ({rows_processed} rows)")
                return True
            else:
                logger.error(f"Verification failed for {parquet_path}: File appears to be empty")
                return False
        except Exception as e:
            logger.error(f"Verification failed for {parquet_path}: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error converting {csv_path}: {str(e)}")
        # If file was partially created, remove it
        if os.path.exists(parquet_path):
            try:
                os.remove(parquet_path)
            except:
                pass
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure verbose logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Prepare input and output directories
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    
    # Find all CSV files
    logger.info(f"Searching for {args.csv_extension} files in {input_dir}...")
    csv_files = find_csv_files(input_dir, args.csv_extension)
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Process each file
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        # Determine the output path
        rel_path = os.path.relpath(csv_file, input_dir)
        parquet_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".parquet")
        
        # Convert the file
        result = convert_csv_to_parquet(
            csv_file,
            parquet_file,
            delimiter=args.delimiter,
            chunk_size=args.chunk_size,
            compression=args.compression,
            overwrite=args.overwrite
        )
        
        if result:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info(f"Conversion complete! {successful} succeeded, {failed} failed")
    
    if successful > 0:
        # Calculate approximate space savings
        original_size = sum(os.path.getsize(f) for f in csv_files if os.path.exists(f))
        converted_files = [os.path.join(output_dir, os.path.splitext(os.path.relpath(f, input_dir))[0] + ".parquet") 
                          for f in csv_files]
        converted_size = sum(os.path.getsize(f) for f in converted_files if os.path.exists(f))
        
        if original_size > 0:
            savings = (1 - (converted_size / original_size)) * 100
            logger.info(f"Space savings: {savings:.1f}% ({converted_size / 1024**2:.1f} MB vs {original_size / 1024**2:.1f} MB)")

if __name__ == "__main__":
    main() 
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
                    engine='c'
                )
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
                    low_memory=True, 
                    encoding='utf-8', 
                    on_bad_lines='warn',
                    engine='c'
                ):
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
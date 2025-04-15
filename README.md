# YouTube Comment Processor

A GPU-accelerated Python tool for cleaning and filtering YouTube comments, specifically developed for the Despacito comments dataset.

## Project Overview

This project provides an efficient solution for processing large YouTube comment datasets by:
1. Filtering out spam content using keyword matching and pattern recognition
2. Detecting and preserving only English-language comments
3. Utilizing GPU acceleration for faster language detection
4. Processing data in parallel across multiple CPU cores

## Main Components

### GPU Comment Processor
- **File:** `gpu_comment_processor.py`
- **Purpose:** Combined spam filtering and language detection in a single workflow
- **Features:**
  - Two-stage filtering (spam removal then language detection)
  - GPU acceleration via Hugging Face Transformers models
  - CPU parallelization for maximum performance
  - Configurable filtering parameters

## Usage

```bash
python gpu_comment_processor.py --input comments.csv --output filtered_comments.csv --use_gpu --keywords spam.txt
```

### Command Line Options:
- `--input`: Path to input CSV file containing comments (required)
- `--output`: Path to output CSV file (default: processed_comments.csv)
- `--comment_col`: Column name containing comments (default: comment)
- `--batch_size`: Batch size for GPU processing (default: 32)
- `--confidence`: Language confidence threshold (default: 0.8)
- `--use_gpu`: Enable GPU acceleration for language detection
- `--cpu_cores`: Number of CPU cores to use (default: all available)
- `--keywords`: File containing spam keywords, one per line (default: spam_keywords.txt)

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- pandas
- NumPy
- CUDA-capable GPU (optional, for GPU acceleration)

## Data

- **Input:** Raw YouTube comments CSV file (`2018.csv`)
- **Output:** Filtered English-only, spam-free comments (`full_processed_comments.csv`)
- **Keywords:** Spam keywords file (`spam_keywords.txt`), one keyword per line

## Performance

The processor efficiently handles large comment datasets by:
- Using GPU acceleration for language detection
- Processing data in parallel across CPU cores
- Filtering in batches to optimize memory usage

This approach significantly reduces processing time compared to traditional CPU-only methods, especially for datasets with millions of comments. 
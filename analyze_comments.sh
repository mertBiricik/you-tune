#!/bin/bash

# You-Tune: YouTube Comment Analyzer Shell Script
# This script provides easy shortcuts for common comment analysis tasks

# Default values
INPUT=""
OUTPUT="analyzed_comments.csv"
USE_GPU=false
ANALYZE_SENTIMENT=false
EXTRACT_TOPICS=false
VISUALIZE=false
COMMENT_COL="comment"
CONFIDENCE=0.8
CPU_CORES=""
BATCH_SIZE=32
N_TOPICS=10
KEYWORDS="spam_keywords.txt"
LANGUAGES=""
LIST_LANGUAGES=false
CONVERT_PARQUET=false
CSV_DIR=""
PARQUET_OUT_DIR=""
CSV_DELIMITER=","
CSV_EXTENSION=".csv"
PARQUET_COMPRESSION="snappy"
OVERWRITE_PARQUET=false

# Help function
show_help() {
    echo "You-Tune: YouTube Comment Analyzer - Easy Runner"
    echo ""
    echo "Usage: ./analyze_comments.sh -i INPUT_FILE [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE       Input CSV file with comments"
    echo "  -o, --output FILE      Output CSV file (default: analyzed_comments.csv)"
    echo "  -c, --column NAME      Column name containing comments (default: comment)"
    echo "  -g, --gpu              Use GPU acceleration if available"
    echo "  -s, --sentiment        Perform sentiment analysis (English comments only)"
    echo "  -t, --topics [N]       Extract topics (optional: specify number, default: 10)"
    echo "  -v, --visualize        Generate visualizations"
    echo "  -k, --keywords FILE    Custom spam keywords file (default: spam_keywords.txt)"
    echo "  -p, --processors N     Number of CPU cores to use"
    echo "  -b, --batch N          Batch size for GPU processing (default: 32)"
    echo "  -l, --languages LANGS  Comma-separated list of language codes to keep (e.g., 'en,es,fr')"
    echo "  -L, --list-languages   List supported language codes and exit"
    echo ""
    echo "CSV to Parquet conversion:"
    echo "  --csv-to-parquet DIR   Convert all CSV files in DIR to Parquet format (recursive)"
    echo "  --parquet-out DIR      Output directory for Parquet files (optional)"
    echo "  --csv-delimiter DELIM  CSV delimiter character (default: ,)"
    echo "  --csv-extension EXT    File extension for CSV files (default: .csv)"
    echo "  --parquet-compression  Compression algorithm [snappy|gzip|brotli|zstd|none] (default: snappy)"
    echo "  --overwrite-parquet    Overwrite existing Parquet files"
    echo ""
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./analyze_comments.sh -i comments.csv -g -s -v -t 15"
    echo "  ./analyze_comments.sh -i comments.csv -l en,es,fr -v -t"
    echo "  ./analyze_comments.sh --csv-to-parquet data_dir --parquet-out parquet_dir"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT="$2"
            shift
            shift
            ;;
        -c|--column)
            COMMENT_COL="$2"
            shift
            shift
            ;;
        -g|--gpu)
            USE_GPU=true
            shift
            ;;
        -s|--sentiment)
            ANALYZE_SENTIMENT=true
            shift
            ;;
        -t|--topics)
            EXTRACT_TOPICS=true
            # Check if next argument is a number
            if [[ $2 =~ ^[0-9]+$ ]]; then
                N_TOPICS=$2
                shift
            fi
            shift
            ;;
        -v|--visualize)
            VISUALIZE=true
            shift
            ;;
        -k|--keywords)
            KEYWORDS="$2"
            shift
            shift
            ;;
        -l|--languages)
            LANGUAGES="$2"
            shift
            shift
            ;;
        -L|--list-languages)
            LIST_LANGUAGES=true
            shift
            ;;
        -p|--processors)
            CPU_CORES="$2"
            shift
            shift
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --csv-to-parquet)
            CONVERT_PARQUET=true
            CSV_DIR="$2"
            shift
            shift
            ;;
        --parquet-out)
            PARQUET_OUT_DIR="$2"
            shift
            shift
            ;;
        --csv-delimiter)
            CSV_DELIMITER="$2"
            shift
            shift
            ;;
        --csv-extension)
            CSV_EXTENSION="$2"
            shift
            shift
            ;;
        --parquet-compression)
            PARQUET_COMPRESSION="$2"
            shift
            shift
            ;;
        --overwrite-parquet)
            OVERWRITE_PARQUET=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# If list languages flag is set, just run with that option and exit
if [ "$LIST_LANGUAGES" = true ]; then
    python you-tune.py --list_languages
    exit 0
fi

# If CSV to Parquet conversion is requested
if [ "$CONVERT_PARQUET" = true ]; then
    if [ -z "$CSV_DIR" ]; then
        echo "ERROR: CSV directory is required for conversion"
        show_help
        exit 1
    fi
    
    # Check if directory exists
    if [ ! -d "$CSV_DIR" ]; then
        echo "ERROR: Directory '$CSV_DIR' does not exist"
        exit 1
    fi
    
    # Build conversion command
    CONVERT_CMD="python csv_to_parquet.py --input-dir \"$CSV_DIR\" --delimiter \"$CSV_DELIMITER\" --csv-extension \"$CSV_EXTENSION\" --compression \"$PARQUET_COMPRESSION\""
    
    # Add output directory if specified
    if [ ! -z "$PARQUET_OUT_DIR" ]; then
        CONVERT_CMD="$CONVERT_CMD --output-dir \"$PARQUET_OUT_DIR\""
    fi
    
    # Add overwrite flag if specified
    if [ "$OVERWRITE_PARQUET" = true ]; then
        CONVERT_CMD="$CONVERT_CMD --overwrite"
    fi
    
    # Add verbose flag for better feedback
    CONVERT_CMD="$CONVERT_CMD --verbose"
    
    # Print and execute command
    echo "Running: $CONVERT_CMD"
    eval $CONVERT_CMD
    exit $?
fi

# For standard comment analysis mode, input file is required
if [ -z "$INPUT" ]; then
    echo "ERROR: Input file is required"
    show_help
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file '$INPUT' does not exist"
    exit 1
fi

# Build command
CMD="python you-tune.py --input \"$INPUT\" --output \"$OUTPUT\" --comment_col \"$COMMENT_COL\" --confidence $CONFIDENCE --keywords \"$KEYWORDS\" --batch_size $BATCH_SIZE"

# Add optional flags
if [ "$USE_GPU" = true ]; then
    CMD="$CMD --use_gpu"
fi

if [ "$ANALYZE_SENTIMENT" = true ]; then
    CMD="$CMD --analyze_sentiment"
fi

if [ "$EXTRACT_TOPICS" = true ]; then
    CMD="$CMD --extract_topics --n_topics $N_TOPICS"
fi

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

if [ ! -z "$CPU_CORES" ]; then
    CMD="$CMD --cpu_cores $CPU_CORES"
fi

if [ ! -z "$LANGUAGES" ]; then
    CMD="$CMD --languages \"$LANGUAGES\""
fi

# Print and execute command
echo "Running: $CMD"
eval $CMD 
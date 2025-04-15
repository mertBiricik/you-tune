import pandas as pd
import re
import os
import time
import multiprocessing
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
from functools import partial
import warnings

# Suppress specific pandas deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*swapaxes.*')

# Spam detection regex patterns
repeat_letters = re.compile(r'(.)\1{3,}')       # 4+ repeated letters
repeat_words = re.compile(r'\b(\w+)\s+\1\b')    # repeated word (e.g., "good good")
repeat_sentences = re.compile(r'(.+?)\.\s+\1\.')  # repeated sentence (e.g., "nice. nice.")

def load_spam_keywords(keywords_file):
    """Load spam keywords from a text file, one keyword per line."""
    try:
        with open(keywords_file, 'r') as f:
            keywords = [line.strip().lower() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(keywords)} spam keywords from {keywords_file}")
        return keywords
    except FileNotFoundError:
        print(f"[ERROR] Spam keywords file '{keywords_file}' not found!")
        print("[WARNING] Using empty spam keywords list. No keyword-based filtering will be applied.")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load spam keywords: {str(e)}")
        print("[WARNING] Using empty spam keywords list. No keyword-based filtering will be applied.")
        return []

def setup_language_model(device):
    """Initialize the language detection model."""
    print(f"[INFO] Setting up language detection model on {device}")
    
    # Load pre-trained language identification model from Hugging Face
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    model = model.to(device)
    
    return model, tokenizer

def detect_language(text, model, tokenizer, device):
    """Detect language for a single text using GPU."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "und", 0.0  # Unknown language for very short or non-string texts
    
    # Tokenize
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted language
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(predictions, dim=1).item()
    score = torch.max(predictions, dim=1).values.item()
    
    # Convert to language code
    id2label = model.config.id2label
    lang = id2label[label_id]
    
    return lang, score

def detect_languages_batch(texts, model, tokenizer, device, batch_size=32):
    """Detect languages for a batch of texts using GPU."""
    results = []
    
    # Process in batches to avoid GPU memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Skip empty texts
        batch_texts = [text if isinstance(text, str) else "" for text in batch_texts]
        batch_texts = [text if len(text.strip()) > 0 else "empty" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted languages
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        labels = torch.argmax(predictions, dim=1)
        scores = torch.max(predictions, dim=1).values
        
        # Convert to language codes
        id2label = model.config.id2label
        batch_langs = [id2label[label_id.item()] for label_id in labels]
        batch_scores = [score.item() for score in scores]
        
        # Combine results
        for lang, score, text in zip(batch_langs, batch_scores, batch_texts):
            results.append((lang, score))
    
    return results

def clean_text(text, spam_keywords):
    """Clean the text by removing spam content."""
    # Skip empty texts
    if not isinstance(text, str) or len(text.strip()) < 3:
        return None
    
    text = text.lower()
    
    # Remove spam-like content
    if any(spam in text for spam in spam_keywords):
        return None
    if repeat_letters.search(text) or repeat_words.search(text) or repeat_sentences.search(text):
        return None
    
    return text

def process_batch(args, model=None, tokenizer=None, device=None, lang_confidence=0.8, spam_keywords=None):
    """Process a batch of comments by cleaning and language filtering."""
    batch, process_id = args
    
    # Initialize language detection model if not provided
    if model is None and device is not None:
        model, tokenizer = setup_language_model(device)
    
    results = []
    total = len(batch)
    kept = 0
    
    # First clean texts to remove spam
    cleaned_texts = []
    clean_indices = []
    
    for idx, row in batch.iterrows():
        cleaned = clean_text(row["comment"], spam_keywords)
        if cleaned is not None:
            cleaned_texts.append(cleaned)
            clean_indices.append(idx)
    
    # Then detect languages (if model is available)
    if model is not None and tokenizer is not None:
        lang_results = detect_languages_batch(cleaned_texts, model, tokenizer, device)
        
        # Filter by language
        for i, (lang, score) in enumerate(lang_results):
            if lang == "en" and score >= lang_confidence:
                idx = clean_indices[i]
                row = batch.iloc[batch.index.get_loc(idx)].copy()
                row["cleaned_text"] = cleaned_texts[i]
                results.append(row)
                kept += 1
    else:
        # If no language model, just keep cleaned texts
        for i, cleaned in enumerate(cleaned_texts):
            idx = clean_indices[i]
            row = batch.iloc[batch.index.get_loc(idx)].copy()
            row["cleaned_text"] = cleaned
            results.append(row)
            kept += 1
    
    # Print progress from each process
    print(f"[INFO-{process_id}] Processed {total} comments, kept {kept} ({kept/total*100:.1f}%)")
    
    # Return dataframe with results
    df_results = pd.DataFrame(results) if results else pd.DataFrame()
    return df_results

def main():
    parser = argparse.ArgumentParser(description="Process comments: clean spam and filter by language")
    parser.add_argument("--input", required=True, help="Input CSV file with comments")
    parser.add_argument("--output", default="processed_comments.csv", help="Output CSV file")
    parser.add_argument("--comment_col", default="comment", help="Column name containing comments")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU processing")
    parser.add_argument("--confidence", type=float, default=0.8, help="Language confidence threshold (0-1)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for language detection")
    parser.add_argument("--cpu_cores", type=int, default=None, help="Number of CPU cores to use (default: all)")
    parser.add_argument("--keywords", default="spam_keywords.txt", help="File containing spam keywords, one per line")
    args = parser.parse_args()
    
    start_time = time.time()
    print("[INFO] Starting comment processing")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!")
        return
    
    # Load spam keywords
    spam_keywords = load_spam_keywords(args.keywords)
    
    # Check if CUDA is available when GPU is requested
    device = None
    model = None
    tokenizer = None
    
    if args.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] Using GPU: {gpu_name} for language detection")
            model, tokenizer = setup_language_model(device)
        else:
            print("[WARNING] No GPU found, falling back to CPU for language detection")
            device = "cpu"
            model, tokenizer = setup_language_model(device)
    
    # Load the comments
    print(f"[INFO] Loading comments from {args.input}")
    df = pd.read_csv(args.input)
    
    # Ensure comment column exists
    if args.comment_col not in df.columns:
        print(f"[ERROR] Comment column '{args.comment_col}' not found in the input file!")
        print(f"[INFO] Available columns: {', '.join(df.columns)}")
        return
    
    # Rename comment column for consistency
    df = df.rename(columns={args.comment_col: "comment"})
    total_comments = len(df)
    print(f"[INFO] Loaded {total_comments} comments")
    
    # Determine number of cores to use
    num_cores = args.cpu_cores or multiprocessing.cpu_count()
    print(f"[INFO] Using {num_cores} CPU cores for processing")
    
    # Split dataframe into chunks for parallel processing
    chunks = np.array_split(df, num_cores)
    
    # Prepare args with process IDs
    batch_args = [(chunk, i) for i, chunk in enumerate(chunks)]
    
    # Process in parallel
    print("[INFO] Starting parallel processing")
    
    if args.use_gpu:
        # For GPU processing, we process chunks sequentially but use GPU for language detection
        results = []
        for chunk_args in batch_args:
            chunk_result = process_batch(
                chunk_args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                lang_confidence=args.confidence,
                spam_keywords=spam_keywords
            )
            results.append(chunk_result)
    else:
        # For CPU-only processing, we use multiprocessing
        with multiprocessing.Pool(processes=num_cores) as pool:
            process_func = partial(process_batch, 
                                  lang_confidence=args.confidence, 
                                  spam_keywords=spam_keywords)
            results = pool.map(process_func, batch_args)
    
    # Combine results
    processed_df = pd.concat(results) if results else pd.DataFrame()
    
    # Save results
    processed_df.to_csv(args.output, index=False)
    
    # Print statistics
    end_time = time.time()
    kept_comments = len(processed_df)
    removed_comments = total_comments - kept_comments
    
    print(f"[INFO] Processing complete in {end_time - start_time:.2f} seconds")
    print(f"[INFO] Original comments: {total_comments}")
    print(f"[INFO] Kept comments: {kept_comments}")
    print(f"[INFO] Removed comments: {removed_comments}")
    print(f"[INFO] Removal rate: {(removed_comments / total_comments * 100):.2f}%")
    print(f"[INFO] Saved processed comments to '{args.output}'")

if __name__ == "__main__":
    main() 
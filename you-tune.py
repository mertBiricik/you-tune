#!/usr/bin/env python3
"""
You-Tune: Unified YouTube Data Processing Toolkit

A comprehensive command-line tool for gathering, processing, analyzing, and visualizing
YouTube comment data. This tool combines functionality from multiple scripts into a 
single interface.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import multiprocessing
import warnings
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("you-tune")

# Suppress specific pandas deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*swapaxes.*')

# Optional imports that will be tried when needed
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

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
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        logger.error("Required packages not installed. Please install with 'pip install torch transformers'")
        return None, None
        
    logger.info(f"Setting up language detection model on {device}")
    
    # Load pre-trained language identification model from Hugging Face
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    model = model.to(device)
    
    return model, tokenizer

def setup_sentiment_model(device):
    """Initialize the sentiment analysis model."""
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers package not installed. Please install with 'pip install transformers'")
        return None
        
    logger.info(f"Setting up sentiment analysis model on {device}")
    
    # Load pre-trained sentiment analysis model from Hugging Face
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Create sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if device == "cuda" else -1  # Use GPU if available
    )
    
    return sentiment_analyzer

def detect_language(text, model, tokenizer, device):
    """Detect language for a single text using GPU."""
    if not TORCH_AVAILABLE or model is None or tokenizer is None:
        return "und", 0.0
        
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
    if not TORCH_AVAILABLE or model is None or tokenizer is None:
        return [(None, 0.0) for _ in texts]
        
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

def get_supported_languages():
    """Get a list of supported languages from the model."""
    # This is a dictionary of language codes to names supported by the model
    return {
        "ar": "Arabic",
        "bg": "Bulgarian", 
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "sw": "Swahili",
        "th": "Thai",
        "tr": "Turkish",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh": "Chinese",
        # This is not a complete list, the actual model supports more languages
    }

def analyze_sentiment_batch(texts, sentiment_analyzer, batch_size=32):
    """Analyze sentiment for a batch of texts."""
    if not TRANSFORMERS_AVAILABLE or sentiment_analyzer is None:
        return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]
        
    results = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Skip empty texts
        batch_texts = [text if isinstance(text, str) else "" for text in batch_texts]
        batch_texts = [text if len(text.strip()) > 0 else "empty" for text in batch_texts]
        
        # Analyze sentiment
        sentiments = sentiment_analyzer(batch_texts)
        
        # Combine results
        results.extend(sentiments)
    
    return results

def extract_topics(texts, n_topics=10):
    """Extract common topics/themes from texts."""
    # Simple word frequency analysis instead of more complex topic modeling
    all_words = []
    
    # Process each text
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) < 3:
            continue
            
        # Basic text cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove common English stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                     'when', 'where', 'how', 'all', 'with', 'is', 'are', 'was', 'were', 'be',
                     'have', 'has', 'had', 'do', 'does', 'did', 'of', 'for', 'by', 'to', 'in',
                     'at', 'on', 'this', 'that', 'these', 'those', 'it', 'its', 'their', 'they'}
        
        # Get words from text
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get top topics
    topics = word_counts.most_common(n_topics)
    
    return topics

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

def process_batch(args, model=None, tokenizer=None, sentiment_analyzer=None, device=None, 
                 lang_confidence=0.8, spam_keywords=None, analyze_sentiment=False,
                 target_languages=None):
    """Process a batch of comments by cleaning, language filtering, and sentiment analysis."""
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
        suitable_texts = []
        suitable_indices = []
        
        for i, (lang, score) in enumerate(lang_results):
            # Accept any language in target_languages with sufficient confidence
            if (not target_languages or lang in target_languages) and score >= lang_confidence:
                idx = clean_indices[i]
                row = batch.iloc[batch.index.get_loc(idx)].copy()
                row["cleaned_text"] = cleaned_texts[i]
                row["language"] = lang
                row["language_confidence"] = score
                
                # For sentiment analysis (only for English)
                if analyze_sentiment and lang == "en":
                    suitable_texts.append(cleaned_texts[i])
                    suitable_indices.append(len(results))
                
                results.append(row)
                kept += 1
                
        # Perform sentiment analysis if requested and model available (English only)
        if analyze_sentiment and sentiment_analyzer and suitable_texts:
            sentiment_results = analyze_sentiment_batch(suitable_texts, sentiment_analyzer)
            
            # Add sentiment scores to results
            for i, sentiment in enumerate(sentiment_results):
                result_idx = suitable_indices[i]
                results[result_idx]["sentiment"] = sentiment["label"]
                results[result_idx]["sentiment_score"] = sentiment["score"]
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

def generate_visualizations(df, output_dir):
    """Generate visualizations from processed comments."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"[INFO] Generating visualizations in {viz_dir}")
    
    # 1. Sentiment Distribution Pie Chart (only for English comments)
    if "sentiment" in df.columns:
        sentiment_df = df[df["language"] == "en"]
        if not sentiment_df.empty:
            plt.figure(figsize=(10, 6))
            sentiment_counts = sentiment_df["sentiment"].value_counts()
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Sentiment Distribution (English Comments)')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "sentiment_distribution.png"))
            plt.close()
            print(f"[INFO] Generated sentiment distribution chart")
    
    # 2. Language Distribution Pie Chart
    if "language" in df.columns:
        plt.figure(figsize=(12, 8))
        language_counts = df["language"].value_counts()
        # Get language names for the most common languages
        language_map = get_supported_languages()
        language_labels = [f"{lang} ({language_map.get(lang, lang)})" for lang in language_counts.index]
        
        plt.pie(language_counts, labels=language_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Language Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "language_distribution.png"))
        plt.close()
        print(f"[INFO] Generated language distribution chart")
    
    # 3. Word Cloud of Comments (per language)
    if "cleaned_text" in df.columns and len(df) > 0 and "language" in df.columns:
        # Get top 5 languages
        top_languages = df["language"].value_counts().head(5).index
        
        for lang in top_languages:
            lang_df = df[df["language"] == lang]
            if not lang_df.empty:
                all_text = " ".join(lang_df["cleaned_text"].dropna())
                if all_text:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
                    plt.figure(figsize=(16, 8))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    language_name = get_supported_languages().get(lang, lang)
                    plt.title(f"Word Cloud - {language_name}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f"wordcloud_{lang}.png"))
                    plt.close()
                    print(f"[INFO] Generated word cloud for {language_name}")
    
    # 4. Top Topics Bar Chart (per language)
    if "cleaned_text" in df.columns and len(df) > 0 and "language" in df.columns:
        # Get top 3 languages
        top_languages = df["language"].value_counts().head(3).index
        
        for lang in top_languages:
            lang_df = df[df["language"] == lang]
            if not lang_df.empty:
                texts = lang_df["cleaned_text"].dropna().tolist()
                topics = extract_topics(texts, n_topics=15)
                
                if topics:
                    topics_df = pd.DataFrame(topics, columns=["Topic", "Count"])
                    plt.figure(figsize=(12, 8))
                    plt.bar(topics_df["Topic"], topics_df["Count"])
                    language_name = get_supported_languages().get(lang, lang)
                    plt.title(f"Most Common Topics - {language_name}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f"topics_{lang}.png"))
                    plt.close()
                    print(f"[INFO] Generated top topics chart for {language_name}")
    
    # 5. If timestamp data is available, create time-based analysis
    if "timestamp" in df.columns:
        try:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            # Create time series of comment frequency
            df_time = df.set_index("timestamp")
            df_time = df_time.resample("D").size()
            
            plt.figure(figsize=(14, 6))
            df_time.plot()
            plt.title("Comment Frequency Over Time")
            plt.xlabel("Date")
            plt.ylabel("Number of Comments")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "comment_time_series.png"))
            plt.close()
            print(f"[INFO] Generated time-based analysis")
        except Exception as e:
            print(f"[WARNING] Could not generate time-based analysis: {str(e)}")
    
    return viz_dir

def analyze_comments(args):
    """Main function to analyze YouTube comments."""
    start_time = time.time()
    print("[INFO] Starting comment analysis")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' not found!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    
    # Load spam keywords
    spam_keywords = load_spam_keywords(args.keywords)
    
    # Process target languages
    target_languages = None
    if args.languages:
        target_languages = [lang.strip().lower() for lang in args.languages.split(',')]
        supported = get_supported_languages()
        print(f"[INFO] Filtering for languages: {', '.join(target_languages)}")
        
        # Validate languages
        for lang in target_languages:
            if lang not in supported:
                print(f"[WARNING] Language '{lang}' may not be supported by the model")
    
    # If list_languages flag is set, print supported languages and exit
    if args.list_languages:
        supported = get_supported_languages()
        print("\nSupported Languages:")
        print("--------------------")
        for code, name in supported.items():
            print(f"{code}: {name}")
        print("\nNote: The model may support additional languages not listed here.")
        return
    
    # Check if CUDA is available when GPU is requested
    device = None
    lang_model = None
    tokenizer = None
    sentiment_analyzer = None
    
    if args.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] Using GPU: {gpu_name} for processing")
            lang_model, tokenizer = setup_language_model(device)
            if args.analyze_sentiment:
                sentiment_analyzer = setup_sentiment_model(device)
        else:
            print("[WARNING] No GPU found, falling back to CPU")
            device = "cpu"
            lang_model, tokenizer = setup_language_model(device)
            if args.analyze_sentiment:
                sentiment_analyzer = setup_sentiment_model(device)
    
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
                model=lang_model,
                tokenizer=tokenizer,
                sentiment_analyzer=sentiment_analyzer,
                device=device,
                lang_confidence=args.confidence,
                spam_keywords=spam_keywords,
                analyze_sentiment=args.analyze_sentiment,
                target_languages=target_languages
            )
            results.append(chunk_result)
    else:
        # For CPU-only processing, we use multiprocessing
        with multiprocessing.Pool(processes=num_cores) as pool:
            process_func = partial(process_batch, 
                                  lang_confidence=args.confidence, 
                                  spam_keywords=spam_keywords,
                                  analyze_sentiment=args.analyze_sentiment,
                                  target_languages=target_languages)
            results = pool.map(process_func, batch_args)
    
    # Combine results
    processed_df = pd.concat(results) if results else pd.DataFrame()
    
    # Extract topics if requested
    if args.extract_topics and not processed_df.empty and "cleaned_text" in processed_df.columns:
        print("[INFO] Extracting topics from comments")
        
        # Group by language if multiple languages
        if "language" in processed_df.columns and len(processed_df["language"].unique()) > 1:
            metadata = {"languages": {}}
            
            # Get top languages by count
            top_languages = processed_df["language"].value_counts().head(5).index
            
            for lang in top_languages:
                lang_df = processed_df[processed_df["language"] == lang]
                texts = lang_df["cleaned_text"].dropna().tolist()
                topics = extract_topics(texts, n_topics=args.n_topics)
                
                lang_name = get_supported_languages().get(lang, lang)
                metadata["languages"][lang] = {
                    "name": lang_name,
                    "topics": [{"topic": topic, "count": int(count)} for topic, count in topics]
                }
        else:
            # Just extract overall topics
            texts = processed_df["cleaned_text"].dropna().tolist()
            topics = extract_topics(texts, n_topics=args.n_topics)
            metadata = {
                "topics": [{"topic": topic, "count": int(count)} for topic, count in topics]
            }
        
        # Save metadata to JSON
        metadata_file = os.path.splitext(args.output)[0] + "_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Saved topic metadata to '{metadata_file}'")
    
    # Generate visualizations if requested
    if args.visualize and not processed_df.empty:
        viz_dir = generate_visualizations(processed_df, os.path.dirname(os.path.abspath(args.output)) or ".")
        print(f"[INFO] Visualizations saved to: {viz_dir}")
    
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
    
    # Print language statistics if multiple languages
    if "language" in processed_df.columns and len(processed_df["language"].unique()) > 1:
        print("\nLanguage Distribution:")
        lang_counts = processed_df["language"].value_counts()
        lang_map = get_supported_languages()
        
        for lang, count in lang_counts.items():
            lang_name = lang_map.get(lang, lang)
            percentage = (count / kept_comments) * 100
            print(f"  {lang} ({lang_name}): {count} comments ({percentage:.1f}%)")

def label_comments(args):
    """Label comments for specified criteria (e.g., tourism-related)."""
    logger.info(f"Labeling comments in {args.input}")
    
    # Define labeling criteria based on type
    if args.type.lower() == 'tourism':
        # Tourism/destination marketing keywords
        keywords = [
            'puerto rico', 'visit', 'vacation', 'tourist', 'tourism', 'travel', 
            'destination', 'beach', 'island', 'resort', 'hotel', 'trip', 'visit puerto rico',
            'beautiful place', 'san juan', 'caribbean', 'beautiful country', 'paradise',
            'place to visit', 'vacation spot', 'landscape', 'scenery', 'venue', 'location',
            'la perla', 'old san juan'
        ]
        
        logger.info(f"Using tourism criteria with {len(keywords)} keywords")
    else:
        logger.error(f"Unsupported labeling type: {args.type}")
        logger.info("Supported types: tourism")
        return
    
    # Function to check if a comment matches the criteria
    def matches_criteria(comment):
        if not isinstance(comment, str):
            return False
            
        comment_lower = comment.lower()
        
        # Check for keywords
        for keyword in keywords:
            if keyword in comment_lower:
                return True
        
        return False
    
    # Load the input file
    logger.info(f"Reading {args.input}...")
    
    try:
        # Get total rows for information purposes
        total_rows = sum(1 for _ in open(args.input)) - 1
        logger.info(f"Total rows in file: {total_rows}")
        
        if args.sample:
            # Take a random sample
            sample_size = min(args.sample, total_rows)
            logger.info(f"Taking a sample of approximately {sample_size} rows...")
            df = pd.read_csv(args.input)
            df = df.sample(n=sample_size, random_state=42)
        else:
            # Process the entire file
            df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    # Apply the labeling function
    logger.info(f"Labeling {len(df)} comments...")
    
    df[f'{args.type}_related'] = df['comment'].apply(matches_criteria).astype(int)
    
    # Set default output filename if not provided
    if not args.output:
        base_name = os.path.basename(args.input)
        args.output = f"labeled_{base_name}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the labeled data
    df.to_csv(args.output, index=False)
    logger.info(f"Labeled data saved to {args.output}")
    
    # Print summary statistics
    label_count = df[f'{args.type}_related'].sum()
    logger.info(f"{args.type.title()}-related comments: {label_count} ({label_count/len(df)*100:.2f}%)")
    logger.info(f"Non-{args.type}-related comments: {len(df) - label_count} ({(len(df) - label_count)/len(df)*100:.2f}%)")

def analyze_sentiment(args):
    """Analyze sentiment of comments."""
    logger.info(f"Analyzing sentiment of comments in {args.input}")
    
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("Transformers library not installed. Please install with 'pip install transformers'")
        return
    
    # Load the input file
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    # Ensure the comment column exists
    if args.comment_column not in df.columns:
        logger.error(f"Comment column '{args.comment_column}' not found in the input file")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Initialize sentiment analyzer
    logger.info("Initializing sentiment analyzer...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Function to truncate comment to a reasonable length
    def truncate_comment(comment, max_length=500):
        if pd.isna(comment) or isinstance(comment, float):
            return ""
        comment = str(comment)
        if len(comment) > max_length:
            return comment[:max_length] + "..."
        return comment
    
    # Get comments as list and ensure they are strings
    comments = df[args.comment_column].fillna('').astype(str).tolist()
    
    # Analyze sentiment in batches
    logger.info(f"Analyzing sentiment for {len(comments)} comments...")
    sentiments = []
    batch_size = args.batch_size
    
    # Process comments in batches
    for i in tqdm(range(0, len(comments), batch_size), desc="Analyzing sentiment"):
        batch = comments[i:i + batch_size]
        # Ensure each comment is a string and truncate if too long
        batch = [truncate_comment(comment) for comment in batch]
        
        try:
            results = sentiment_analyzer(batch, truncation=True, max_length=512)
            sentiments.extend(results)
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {str(e)}")
            # Add default sentiment for failed batch
            sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5} for _ in batch])
    
    # Add sentiment columns to dataframe
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    
    # Set default output filename if not provided
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '_with_sentiment.csv'
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print summary statistics
    sentiment_counts = df['sentiment_label'].value_counts()
    logger.info("\nSentiment Distribution:")
    for label, count in sentiment_counts.items():
        logger.info(f"  {label}: {count} ({count/len(df)*100:.2f}%)")

def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="You-Tune: Unified YouTube Data Processing Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for main command categories
    subparsers = parser.add_subparsers(dest="command", help="Command category")
    
    # Conversion commands
    convert_parser = subparsers.add_parser("convert", help="Data conversion operations")
    convert_subparsers = convert_parser.add_subparsers(dest="subcommand", help="Conversion operation")
    
    # CSV to Parquet
    csv2parquet_parser = convert_subparsers.add_parser("csv2parquet", help="Convert CSV to Parquet format")
    csv2parquet_parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing CSV files")
    csv2parquet_parser.add_argument("--output-dir", "-o", help="Output directory for Parquet files")
    csv2parquet_parser.add_argument("--delimiter", "-d", default=",", help="CSV delimiter (default: comma)")
    csv2parquet_parser.add_argument("--chunk-size", "-c", type=int, default=100000, help="Chunk size for processing")
    csv2parquet_parser.add_argument("--compression", choices=["snappy", "gzip", "brotli", "zstd", "none"], default="snappy", help="Compression algorithm")
    csv2parquet_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    csv2parquet_parser.set_defaults(func=convert_csv_to_parquet)
    
    # JSON to CSV
    json2csv_parser = convert_subparsers.add_parser("json2csv", help="Convert JSON to CSV format")
    json2csv_parser.add_argument("input_file", help="Input JSON file")
    json2csv_parser.add_argument("--output-file", "-o", help="Output CSV file")
    json2csv_parser.set_defaults(func=convert_json_to_csv)
    
    # Gathering commands
    gather_parser = subparsers.add_parser("gather", help="Data gathering operations")
    gather_subparsers = gather_parser.add_subparsers(dest="subcommand", help="Gathering operation")
    
    # Gather YouTube comments
    comments_parser = gather_subparsers.add_parser("comments", help="Gather YouTube comments")
    comments_parser.add_argument("--video-id", "-v", required=True, help="YouTube video ID")
    comments_parser.add_argument("--output-file", "-o", default="youtube_comments.csv", help="Output CSV file")
    comments_parser.add_argument("--limit", "-l", type=int, help="Maximum number of comments to gather")
    comments_parser.add_argument("--api-key", "-k", help="YouTube API key (can be multiple keys separated by comma)")
    comments_parser.set_defaults(func=gather_comments)
    
    # Processing commands
    process_parser = subparsers.add_parser("process", help="Data processing operations")
    process_subparsers = process_parser.add_subparsers(dest="subcommand", help="Processing operation")
    
    # Filter spam and language
    filter_parser = process_subparsers.add_parser("filter", help="Filter spam and non-English comments")
    filter_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    filter_parser.add_argument("--output", "-o", default="filtered_comments.csv", help="Output CSV file")
    filter_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for language detection")
    filter_parser.add_argument("--cpu-cores", type=int, help="Number of CPU cores to use")
    filter_parser.set_defaults(func=filter_spam_and_language)
    
    # Split by year
    split_parser = process_subparsers.add_parser("split-by-year", help="Split data by year")
    split_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    split_parser.add_argument("--date-column", "-d", required=True, help="Column containing date")
    split_parser.add_argument("--output-dir", "-o", default="yearly_data", help="Output directory")
    split_parser.add_argument("--date-format", "-f", help="Date format (e.g., '%%Y-%%m-%%d')")
    split_parser.set_defaults(func=split_by_year)
    
    # Label comments
    label_parser = process_subparsers.add_parser("label", help="Label comments")
    label_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    label_parser.add_argument("--output", "-o", help="Output CSV file")
    label_parser.add_argument("--type", "-t", default="tourism", help="Labeling type (tourism, etc.)")
    label_parser.add_argument("--sample", "-s", type=int, help="Sample size to process")
    label_parser.set_defaults(func=label_comments)
    
    # Sentiment analysis
    sentiment_parser = process_subparsers.add_parser("sentiment", help="Analyze sentiment of comments")
    sentiment_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    sentiment_parser.add_argument("--output", "-o", help="Output CSV file")
    sentiment_parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for processing")
    sentiment_parser.add_argument("--comment-column", "-c", default="comment", help="Column containing comments")
    sentiment_parser.set_defaults(func=analyze_sentiment)
    
    # Validation commands
    validate_parser = subparsers.add_parser("validate", help="Data validation operations")
    validate_subparsers = validate_parser.add_subparsers(dest="subcommand", help="Validation operation")
    
    # Validate Parquet files
    validate_parquet_parser = validate_subparsers.add_parser("parquet", help="Validate Parquet files")
    validate_parquet_parser.add_argument("--csv", help="Path to CSV file (for single file validation)")
    validate_parquet_parser.add_argument("--parquet", help="Path to Parquet file (for single file validation)")
    validate_parquet_parser.add_argument("--parquet-dir", default="./parquet", help="Directory with Parquet files (for batch validation)")
    validate_parquet_parser.add_argument("--csv-dir", default="./data", help="Directory with CSV files (for batch validation)")
    validate_parquet_parser.add_argument("--output-dir", default="./validation_reports", help="Output directory for reports")
    validate_parquet_parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to compare")
    validate_parquet_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    validate_parquet_parser.set_defaults(func=validate_parquet)
    
    # Examine Parquet file
    examine_parser = validate_subparsers.add_parser("examine", help="Examine Parquet file")
    examine_parser.add_argument("--parquet", required=True, help="Path to Parquet file")
    examine_parser.add_argument("--samples", type=int, default=10, help="Number of sample rows to display")
    examine_parser.set_defaults(func=examine_parquet)
    
    # Visualization commands
    viz_parser = subparsers.add_parser("visualize", help="Data visualization operations")
    viz_subparsers = viz_parser.add_subparsers(dest="subcommand", help="Visualization operation")
    
    # Visualize results
    results_viz_parser = viz_subparsers.add_parser("results", help="Visualize analysis results")
    results_viz_parser.add_argument("--file", "-f", help="Labeled CSV file to analyze")
    results_viz_parser.add_argument("--dir", "-d", help="Directory of labeled CSV files")
    results_viz_parser.set_defaults(func=visualize_results)
    
    # Visualize yearly trends
    trends_viz_parser = viz_subparsers.add_parser("trends", help="Visualize yearly trends")
    trends_viz_parser.add_argument("--input-dir", "-i", required=True, help="Input directory with yearly data")
    trends_viz_parser.add_argument("--output-dir", "-o", default="visualizations", help="Output directory for visualizations")
    trends_viz_parser.add_argument("--label-column", "-l", default="tourism_related", help="Column with labels")
    trends_viz_parser.set_defaults(func=visualize_yearly_trends)
    
    # Parse args and call the appropriate function
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

#============================================================================
# Data Conversion Module
#============================================================================

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

def convert_single_csv_to_parquet(
    csv_path, 
    parquet_path, 
    delimiter=",", 
    chunk_size=100000, 
    compression="snappy", 
    overwrite=False
):
    """Convert a single CSV file to Parquet format."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow not installed. Please install it with 'pip install pyarrow'")
        return False
    
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

def convert_csv_to_parquet(args):
    """Convert CSV files to Parquet format."""
    logger.info("Converting CSV to Parquet...")
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow not installed. Please install it with 'pip install pyarrow'")
        return
    
    # Prepare input and output directories
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    
    # Find all CSV files
    logger.info(f"Searching for CSV files in {input_dir}...")
    csv_files = find_csv_files(input_dir, ".csv")
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Process each file
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        # Determine the output path
        rel_path = os.path.relpath(csv_file, input_dir)
        parquet_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".parquet")
        
        # Convert the file
        result = convert_single_csv_to_parquet(
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

def convert_json_to_csv(args):
    """Convert JSON files to CSV format."""
    logger.info(f"Converting JSON to CSV: {args.input_file}")
    
    try:
        # Open and load the JSON file
        with open(args.input_file) as fp:
            data = json.load(fp)
            
            # Check if it has a 'comments' key (based on the format in json2csv.py)
            if isinstance(data, dict) and 'comments' in data:
                comments = data['comments']
            elif isinstance(data, list):
                comments = data
            else:
                comments = data  # Assume the whole JSON is the data we want
                
            # Determine field names (columns)
            fields = []
            for item in comments:
                for k in item.keys():
                    if k not in fields:
                        fields.append(k)
            
            # Determine output file name
            output_file = args.output_file if args.output_file else os.path.splitext(args.input_file)[0] + '.csv'
            
            # Write to CSV file
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = pd.DataFrame(comments).to_csv(csvfile, index=False)
                
            logger.info(f"Successfully converted {args.input_file} to {output_file}")
            logger.info(f"Found {len(comments)} records with {len(fields)} fields")
            
    except Exception as e:
        logger.error(f"Error converting JSON to CSV: {str(e)}")
        return 
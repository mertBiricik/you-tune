#!/usr/bin/env python3
import pandas as pd
import re
import os
import time
import multiprocessing
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)
import numpy as np
from tqdm import tqdm
import argparse
from functools import partial
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import datetime
import json

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

def setup_sentiment_model(device):
    """Initialize the sentiment analysis model."""
    print(f"[INFO] Setting up sentiment analysis model on {device}")
    
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
    """Extract most common topics from texts using simple word frequency."""
    if not texts:
        return []
    
    # Combine all texts into a single string
    all_text = " ".join(texts)
    
    # Create a vectorizer that removes common English stop words
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=2,  # Minimum document frequency
        ngram_range=(1, 2)  # Consider both unigrams and bigrams
    )
    
    # Get word counts
    X = vectorizer.fit_transform([all_text])
    words = vectorizer.get_feature_names_out()
    counts = X.toarray()[0]
    
    # Get top N words by count
    top_indices = counts.argsort()[-n_topics:][::-1]
    topics = [(words[i], counts[i]) for i in top_indices]
    
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
                    sns.barplot(x="Count", y="Topic", data=topics_df)
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

def main():
    parser = argparse.ArgumentParser(description="You-Tune: YouTube Comment Analyzer")
    
    # Input/Output options
    parser.add_argument("--input", required=True, help="Input CSV file with comments")
    parser.add_argument("--output", default="analyzed_comments.csv", help="Output CSV file")
    parser.add_argument("--comment_col", default="comment", help="Column name containing comments")
    
    # Filtering options
    parser.add_argument("--confidence", type=float, default=0.8, help="Language confidence threshold (0-1)")
    parser.add_argument("--keywords", default="spam_keywords.txt", help="File containing spam keywords, one per line")
    parser.add_argument("--languages", default="", help="Comma-separated list of language codes to keep (e.g., 'en,es,fr'). Empty = all languages")
    parser.add_argument("--list_languages", action="store_true", help="List supported language codes and exit")
    
    # Analysis options
    parser.add_argument("--analyze_sentiment", action="store_true", help="Perform sentiment analysis on comments (English only)")
    parser.add_argument("--extract_topics", action="store_true", help="Extract topics from comments")
    parser.add_argument("--n_topics", type=int, default=10, help="Number of topics to extract")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations from the analysis")
    
    # Performance options
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU processing")
    parser.add_argument("--cpu_cores", type=int, default=None, help="Number of CPU cores to use (default: all)")
    
    args = parser.parse_args()
    analyze_comments(args)

if __name__ == "__main__":
    main() 
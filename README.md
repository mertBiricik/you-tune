# You-Tune: YouTube Comment Analyzer

Command-line YouTube comment analysis tool. Multi-language support, spam filtering, sentiment analysis, topic extraction.

## Current Capabilities

**Language Processing:**
- Language detection: XLM-RoBERTa model
- Support: 20+ languages including English, Spanish, French, German
- Sentiment analysis: English comments only (DistilBERT)

**Spam Detection:**
- Repetitive content filtering
- Customizable keyword blacklists
- Regex pattern matching for spam indicators

**Analysis Features:**
- Topic extraction by language groups
- Visualization generation
- Language distribution analysis
- Performance optimization with GPU acceleration

## Requirements

```bash
pip install torch transformers pandas numpy matplotlib seaborn wordcloud scikit-learn tqdm
```

Python 3.6+ required.

## Installation

Current method:
```bash
git clone [repository]
pip install -r requirements.txt
chmod +x analyze_comments.sh
```

## Usage

**Basic Analysis:**
```bash
python you-tune.py --input comments.csv --output analyzed_comments.csv
```

**Shell Script:**
```bash
./analyze_comments.sh -i comments.csv -o analyzed_comments.csv
```

## File Structure

```
you-tune/
├── you-tune.py             # Main script
├── analyze_comments.sh     # Shell wrapper
├── spam_keywords.txt       # Spam keyword list
└── requirements.txt        # Dependencies
```

## Limitations

**Language Support:**
- Sentiment analysis English-only
- Language detection accuracy varies by text length
- No cultural context preservation for non-English text

**Performance:**
- GPU acceleration optional but recommended
- Memory usage scales with input size
- No batch processing optimization

**Analysis Quality:**
- Topic extraction depends on comment volume
- Spam detection basic pattern matching
- No validation against ground truth datasets

## Technical Constraints

**Dependencies:**
- PyTorch required for language detection
- Transformers package for model inference
- GPU drivers if using CUDA acceleration

**Input Format:**
- CSV files with text column
- UTF-8 encoding required
- Large files may cause memory issues

**Output Quality:**
- Visualization quality depends on data distribution
- Topic coherence varies by language and content
- No quality metrics provided

## Development Status

Active development. Core functionality stable. No backwards compatibility guarantees. 
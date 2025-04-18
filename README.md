# You-Tune: YouTube Comment Analyzer

A command-line utility for analyzing YouTube comments, providing multi-language support, spam filtering, sentiment analysis, topic extraction, and visualization.

## Features

- **Multi-language Support**: Process comments in 20+ languages including English, Spanish, French, German, and more
- **Spam Detection & Filtering**: Remove repetitive content and filter using customizable keywords
- **Language Detection**: Use a pre-trained XLM-RoBERTa model to identify comment languages
- **Sentiment Analysis**: Analyze the sentiment of English comments (positive/negative)
- **Topic Extraction**: Identify key topics and themes in your comments, grouped by language
- **Visualizations**: Generate charts, wordclouds, and language distribution graphs
- **Performance Optimization**: Use GPU acceleration and parallel processing for speed

## Roadmap

You-Tune is under active development with several exciting features planned:

### Coming Soon
- **YouTube API Integration**: Fetch comments directly using the official YouTube API
- **yt-dlp Integration**: Alternative method to retrieve comments and metadata
- **Video Content Analysis**:
  - Download video audio tracks using yt-dlp
  - Transcribe video content using OpenAI's Whisper
  - Analyze video transcriptions alongside comments
- **Combined Analytics**: Compare content from videos with audience comments
- **Publication as pip Package**: Easy installation via `pip install you-tune`

These upcoming features will transform You-Tune from a comment analyzer into a comprehensive YouTube content analysis toolkit.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- scikit-learn

## Installation

### Current Installation

1. Clone this repository or download the script files
2. Install the required dependencies:

```bash
pip install torch transformers pandas numpy matplotlib seaborn wordcloud scikit-learn tqdm
```

3. Make the shell script executable:

```bash
chmod +x analyze_comments.sh
```

### Future pip Installation

When released on PyPI, installation will be as simple as:

```bash
pip install you-tune
```

This will install the package and its dependencies, and make the `you-tune` command available in your PATH.

## Project Structure

```
you-tune/
├── you-tune.py             # Main Python script
├── analyze_comments.sh     # Shell script helper
├── spam_keywords.txt       # Default spam keywords list
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
└── README.md               # This documentation
```

Future package structure (when released on PyPI):

```
you-tune/
├── you_tune/               # Main package directory
│   ├── __init__.py         # Package initialization
│   ├── analyzer.py         # Core analysis functionality
│   ├── fetcher.py          # YouTube comment/content fetching
│   ├── transcriber.py      # Audio transcription utilities
│   ├── visualizer.py       # Visualization tools
│   └── utils/              # Helper utilities
├── scripts/                # Command-line scripts
│   └── you-tune            # Main entry point
├── examples/               # Example usage scripts
├── tests/                  # Test suite
├── setup.py                # Package setup script
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
└── README.md               # Documentation
```

## Contributing

Contributions to You-Tune are welcome! Here's how you can help:

1. **Report bugs and suggest features** by opening issues
2. **Submit pull requests** with bug fixes or new features
3. **Improve documentation** to make the project more accessible
4. **Share your use cases** to help prioritize future development

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/you-tune.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

## Usage

### Command Line Script

Basic usage:

```bash
python you-tune.py --input comments.csv --output analyzed_comments.csv
```

### Shell Script Helper

For easier usage, use the shell script:

```bash
./analyze_comments.sh -i comments.csv -o analyzed_comments.csv
```

### Future pip Package Usage

After installation via pip, you'll be able to use the command directly:

```bash
# Basic usage
you-tune --input comments.csv --output analyzed_comments.csv

# Fetch comments directly from YouTube
you-tune --video-id dQw4w9WgXcQ --output rick_roll_comments.csv

# Download, transcribe and analyze video content
you-tune --video-id dQw4w9WgXcQ --transcribe --analyze-content

# Complete analysis with visualization
you-tune --video-id dQw4w9WgXcQ --transcribe --analyze-content --analyze-sentiment --extract-topics --visualize
```

### YouTube API Integration

The planned YouTube API integration will require:

1. A Google Developer account
2. YouTube Data API v3 credentials
3. Configuration of your API key:

```bash
# Set up your YouTube API key
you-tune --setup-youtube-api

# Alternatively, provide it as an environment variable
export YOUTUBE_API_KEY="your_api_key_here"
you-tune --video-id dQw4w9WgXcQ
```

### yt-dlp Integration

The planned yt-dlp integration will:

1. Download comments without requiring an API key
2. Fetch video metadata
3. Download audio for transcription:

```bash
# Using yt-dlp backend instead of YouTube API
you-tune --video-id dQw4w9WgXcQ --use-ytdlp

# Download and transcribe
you-tune --video-id dQw4w9WgXcQ --use-ytdlp --transcribe
```

### Whisper Transcription

The planned Whisper integration will:

1. Transcribe video audio
2. Support multiple languages
3. Generate timestamped transcripts

```bash
# Basic transcription
you-tune --video-id dQw4w9WgXcQ --transcribe

# Transcription with specific model
you-tune --video-id dQw4w9WgXcQ --transcribe --whisper-model medium

# Transcription with forced language
you-tune --video-id dQw4w9WgXcQ --transcribe --whisper-language en
```

### Command-line Arguments

#### Current Arguments:

#### Input/Output Options:
- `--input`: Input CSV file with comments (required)
- `--output`: Output CSV file (default: "analyzed_comments.csv")
- `--comment_col`: Column name containing comments (default: "comment")

#### Filtering Options:
- `--confidence`: Language confidence threshold, 0-1 (default: 0.8)
- `--keywords`: File containing spam keywords, one per line (default: "spam_keywords.txt")
- `--languages`: Comma-separated list of language codes to keep (e.g., 'en,es,fr'). Empty = all languages
- `--list_languages`: List supported language codes and exit

#### Analysis Options:
- `--analyze_sentiment`: Perform sentiment analysis on English comments
- `--extract_topics`: Extract topics from comments
- `--n_topics`: Number of topics to extract (default: 10)
- `--visualize`: Generate visualizations from the analysis

#### Performance Options:
- `--use_gpu`: Use GPU for processing (if available)
- `--batch_size`: Batch size for GPU processing (default: 32)
- `--cpu_cores`: Number of CPU cores to use (default: all available)

#### Future Arguments (After Roadmap Implementation):

#### YouTube Data Fetching:
- `--video-id`: YouTube video ID to analyze
- `--channel-id`: YouTube channel ID to analyze
- `--max-comments`: Maximum number of comments to fetch (default: 1000)
- `--setup-youtube-api`: Configure YouTube API credentials
- `--use-ytdlp`: Use yt-dlp instead of YouTube API

#### Transcription Options:
- `--transcribe`: Transcribe video audio
- `--whisper-model`: Whisper model to use (tiny, base, small, medium, large)
- `--whisper-language`: Force specific language for transcription
- `--audio-only`: Download only audio without analyzing comments

#### Combined Analysis:
- `--analyze-content`: Analyze transcribed content
- `--compare-content-comments`: Compare video content with comments
- `--sentiment-alignment`: Analyze sentiment alignment between content and comments

## Examples

### List Supported Languages

```bash
python you-tune.py --list_languages
# Or with the shell script:
./analyze_comments.sh -L
```

### Filter for Specific Languages

Process comments in Spanish, English, and French only:

```bash
python you-tune.py --input comments.csv --languages en,es,fr
# Or with the shell script:
./analyze_comments.sh -i comments.csv -l en,es,fr
```

### Basic Comment Filtering

Filter comments for spam in all languages:

```bash
python you-tune.py --input raw_comments.csv --output filtered_comments.csv
# Or with the shell script:
./analyze_comments.sh -i raw_comments.csv -o filtered_comments.csv
```

### Sentiment Analysis with Visualizations

Analyze sentiment and generate visualization charts (for English comments only):

```bash
python you-tune.py --input comments.csv --analyze_sentiment --visualize
# Or with the shell script:
./analyze_comments.sh -i comments.csv -s -v
```

### Topic Extraction by Language

Extract the top 20 topics from comments, grouped by language:

```bash
python you-tune.py --input comments.csv --extract_topics --n_topics 20
# Or with the shell script:
./analyze_comments.sh -i comments.csv -t 20
```

### Using GPU Acceleration

Use GPU for faster processing (if available):

```bash
python you-tune.py --input large_comment_dataset.csv --use_gpu --batch_size 64
# Or with the shell script:
./analyze_comments.sh -i large_comment_dataset.csv -g -b 64
```

### Future Examples (After Roadmap Implementation):

### Fetch and Analyze Comments from a Video

```bash
you-tune --video-id dQw4w9WgXcQ --output rickroll_comments.csv --analyze_sentiment --visualize
```

### Transcribe Video and Analyze Content

```bash
you-tune --video-id dQw4w9WgXcQ --transcribe --analyze-content --extract_topics
```

### Comprehensive Video Analysis

```bash
you-tune --video-id dQw4w9WgXcQ --transcribe --analyze-content --analyze_sentiment --extract_topics --compare-content-comments --visualize
```

## Output

The tool generates:

1. A CSV file with cleaned comments and analysis results including:
   - Original comment text
   - Cleaned text
   - Detected language and confidence score
   - Sentiment analysis (for English comments)

2. A JSON file with extracted topics (if `--extract_topics` is used):
   - Overall topics or topics by language
   - Topic frequency counts

3. Visualizations in a timestamped directory (if `--visualize` is used):
   - Language distribution pie chart
   - Sentiment distribution pie chart (English comments)
   - Word clouds by language
   - Top topics bar charts by language
   - Time series of comment frequency (if timestamp data is available)

## Future Output (After Roadmap Implementation)

When video content analysis features are implemented, the tool will additionally provide:

1. Video transcriptions in text format with timestamps
2. Content analysis comparing video topics with comment topics
3. Sentiment alignment between video content and audience reactions
4. Cross-language insights for videos with international audiences
5. Combined reports with visualizations comparing content and audience engagement

## Creating a Spam Keywords File

Create a text file (default: `spam_keywords.txt`) with one keyword or phrase per line:
```
buy now
discount code
check out my channel
subscribe to
```

## License

[MIT License](LICENSE) 
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="you-tune",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive YouTube comment and content analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/you-tune",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/you-tune/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "you-tune=you_tune.cli:main",
        ],
    },
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "wordcloud>=1.8.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "youtube": [
            "google-api-python-client>=2.0.0",
            "google-auth-oauthlib>=0.4.0",
        ],
        "download": [
            "yt-dlp>=2022.1.21",
        ],
        "transcribe": [
            "openai-whisper>=20230124",
        ],
        "full": [
            "google-api-python-client>=2.0.0",
            "google-auth-oauthlib>=0.4.0",
            "yt-dlp>=2022.1.21",
            "openai-whisper>=20230124",
            "tensorflow>=2.8.0",
            "nltk>=3.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    include_package_data=True,
) 
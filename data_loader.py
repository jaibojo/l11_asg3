import os
import json
from pathlib import Path

def download_hindi_dataset(max_articles=1000, max_chars_per_article=1000):
    """Download Hindi Wikipedia dataset from Kaggle and create a smaller test corpus"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Check if test corpus file exists
        corpus_path = data_dir / 'hindi_corpus_test.txt'
        if corpus_path.exists() and corpus_path.stat().st_size > 0:
            print(f"Using existing test corpus file at {corpus_path}")
            print(f"Corpus size: {corpus_path.stat().st_size/1024:.2f} KB")
            return str(corpus_path)
            
        # Check for downloaded files
        downloaded_files = list(data_dir.glob('*'))
        if downloaded_files:
            print(f"Found existing dataset with {len(downloaded_files)} files")
        else:
            # Setup Kaggle credentials directory
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json = kaggle_dir / 'kaggle.json'
            
            # Create kaggle.json if it doesn't exist
            if not kaggle_json.exists():
                credentials = {
                    "username": "jaigoyal24",
                    "key": "33b9f2599e227fe54cd7714630245fd4"
                }
                
                with open(kaggle_json, 'w') as f:
                    json.dump(credentials, f)
                
                # Set required permissions
                os.chmod(kaggle_json, 0o600)
                print(f"Created Kaggle credentials file at {kaggle_json}")
            
            # Import kaggle after credentials are set
            import kaggle
            
            # Download dataset
            print("Downloading Hindi Wikipedia dataset...")
            kaggle.api.dataset_download_files(
                'disisbig/hindi-wikipedia-articles-172k',
                path='data',
                unzip=True
            )
            print("Dataset downloaded successfully!")
        
        # Find Hindi text files
        text_files = []
        for file in data_dir.rglob('*'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read(100)
                        if any('\u0900' <= c <= '\u097F' for c in content):
                            text_files.append(file)
                except (UnicodeDecodeError, IOError):
                    continue
        
        if not text_files:
            raise ValueError("No Hindi text files found in the dataset!")
            
        print(f"\nFound {len(text_files)} Hindi text files")
        print(f"Creating test corpus with max {max_articles} articles")
        
        # Create smaller test corpus
        total_articles = 0
        total_chars = 0
        
        with open(corpus_path, 'w', encoding='utf-8') as outfile:
            for file in text_files[:max_articles]:  # Limit number of articles
                with open(file, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    if content:
                        # Limit article length
                        content = content[:max_chars_per_article]
                        outfile.write(content + '\n\n')
                        total_articles += 1
                        total_chars += len(content)
        
        # Verify corpus file
        corpus_size = corpus_path.stat().st_size
        print(f"\nCreated test corpus file at {corpus_path}")
        print(f"Corpus size: {corpus_size/1024:.2f} KB")
        print(f"Total articles processed: {total_articles}")
        print(f"Total characters: {total_chars}")
        
        if corpus_size == 0:
            raise ValueError("No data was written to the corpus file!")
            
        return str(corpus_path)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. You have an active internet connection")
        print("2. The dataset is still available on Kaggle")
        print("3. Your Kaggle API quota has not been exceeded")
        return None 
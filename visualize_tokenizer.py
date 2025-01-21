from hindi_tokenizer import HindiTokenizer
from data_loader import download_hindi_dataset
import time

def main():
    # Download and prepare dataset
    corpus_path = download_hindi_dataset()
    if not corpus_path:
        print("Failed to load dataset")
        return
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    start_time = time.time()
    
    tokenizer = HindiTokenizer(corpus_path)
    
    end_time = time.time()
    print(f"\nTokenization completed in {end_time - start_time:.2f} seconds")
    
    # Print compression stats
    compression_ratio = tokenizer.get_compression_ratio()
    print(f"\nCompression ratio: {compression_ratio:.2f}")

if __name__ == "__main__":
    main() 
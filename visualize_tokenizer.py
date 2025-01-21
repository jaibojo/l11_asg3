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
    
    # Get and print statistics
    stats = tokenizer.get_stats()
    print("\nTokenization Statistics:")
    print(f"1. Initial tokens: {stats['initial_tokens']:,}")
    print(f"   Initial vocabulary size: {stats['initial_vocab']:,}")
    print(f"\n2. Final tokens: {stats['final_tokens']:,}")
    print(f"   Final vocabulary size: {stats['final_vocab']:,}")
    print(f"\n3. Compression ratio: {stats['compression_ratio']:.2f}")

if __name__ == "__main__":
    main() 
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from tabulate import tabulate

class HindiTokenizer:
    """
    A tokenizer for Hindi text that uses Byte-Pair Encoding (BPE) on UTF-8 encoded tokens.
    Implements a vocabulary-based compression algorithm that iteratively merges the most
    frequent pairs of UTF-8 codes into new tokens.
    """
    def __init__(self, file_path: str):
        # Basic configuration
        self.file_path = file_path
        self.initial_vocab_size = 0  # Tracks size of initial vocabulary (characters + special tokens)
        self.initial_tokens_length = 0  # Tracks initial number of UTF-8 codes
        self.vocab_size = 50000  # Target final vocabulary size
        
        # Special tokens for various text processing needs
        self.special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<num>', '<eng>']
        
        # Mappings between tokens and their IDs
        self.token_to_id = {}  # Maps tokens to unique IDs
        self.id_to_token = {}  # Maps IDs back to tokens
        self.next_id = 0  # Tracks next available ID for new tokens
        
        try:
            # Step 1 & 2: Load and clean the input text
            self.text = self.load_and_clean_text()
            self.vocab = set()  # Set to store unique tokens
            
            # Step 3: Initialize vocabulary with special tokens and characters
            self.initialize_vocab()
            
            # Step 4: Convert text to UTF-8 encodings for BPE
            self.encoded_tokens = self.convert_to_utf8()
            
            # Store initial statistics
            self.initial_tokens_length = len(self.encoded_tokens)
            self.initial_vocab_size = len(self.vocab)
            
            # Adjust target vocab size to account for initial vocabulary
            self.vocab_size = 50000 - self.initial_vocab_size
            
            # Step 5-7: Build final vocabulary using BPE
            self.build_vocabulary()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
    
    def load_and_clean_text(self) -> str:
        """
        Load text file and clean it by:
        - Replacing numbers with <num> token
        - Replacing English words with <eng> token
        - Removing punctuation and special characters
        - Keeping only Hindi characters and spaces
        """
        # Read large files in chunks to manage memory
        chunk_size = 1024 * 1024  # 1MB chunks
        text = ""
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                text += chunk
        
        # Text cleaning steps
        print("Cleaning text...")
        # Replace numbers with special token
        text = re.sub(r'[०-९0-9]+', ' <num> ', text)
        # Replace English words with special token
        text = re.sub(r'[A-Za-z]+', ' <eng> ', text)
        # Remove punctuation
        text = re.sub(r'[!@#$%^&*(),.?":{}|<>]', ' ', text)
        # Remove Hindi purna viram and double purna viram
        text = re.sub(r'[\u0964\u0965]', ' ', text)
        # Keep only Hindi characters and special tokens
        text = re.sub(r'[^\u0900-\u097F\s<>a-z]', '', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        # Merge consecutive special tokens
        text = re.sub(r'<num>\s*<num>', '<num>', text)
        text = re.sub(r'<eng>\s*<eng>', '<eng>', text)
        
        print("Text cleaning complete!")
        return text.strip()
    
    def add_token(self, token: str):
        """Add a new token to vocabulary with a unique ID"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
            self.vocab.add(token)
    
    def initialize_vocab(self):
        """
        Initialize vocabulary with:
        1. Special tokens (pad, eos, bos, unk, num, eng)
        2. Individual characters from the text
        """
        # Add special tokens first to ensure consistent IDs
        for token in self.special_tokens:
            self.add_token(token)
        
        # Add each unique character from text
        for char in self.text:
            if char.strip():  # Skip whitespace
                self.add_token(char)
    
    def convert_to_utf8(self) -> List[List[int]]:
        """
        Convert text tokens to UTF-8 encodings.
        Returns a list of lists, where each inner list contains
        the UTF-8 codes for a token.
        """
        tokens = self.text.split()
        encoded_tokens = []
        for token in tokens:
            # Convert each character to its UTF-8 encoding
            encoded_token = []
            for char in token:
                encoded_token.extend(char.encode('utf-8'))
            encoded_tokens.append(encoded_token)
        return encoded_tokens
    
    def get_pairs(self) -> Counter:
        """
        Find all adjacent pairs of UTF-8 codes and count their frequencies.
        Used to identify the most common pairs for merging.
        """
        pairs = Counter()
        
        for encoded_token in self.encoded_tokens:
            if len(encoded_token) < 2:
                continue
            
            # Count frequencies of adjacent pairs
            for i in range(len(encoded_token)-1):
                pair = (encoded_token[i], encoded_token[i+1])
                pairs[pair] += 1
        
        return pairs
    
    def merge_pair(self, pair: Tuple[int, int], merged_token: Tuple[int, int], token_id: int) -> bool:
        """
        Replace all occurrences of a UTF-8 code pair with a new token ID.
        Returns True if any merges were performed.
        """
        any_changes = False
        new_encoded_tokens = []
        
        for encoded_token in self.encoded_tokens:
            new_token = []
            i = 0
            while i < len(encoded_token):
                # Check if current position has the target pair
                if (i < len(encoded_token)-1 and 
                    encoded_token[i] == pair[0] and 
                    encoded_token[i+1] == pair[1]):
                    new_token.append(token_id)
                    i += 2
                    any_changes = True
                else:
                    new_token.append(encoded_token[i])
                    i += 1
            new_encoded_tokens.append(new_token)
        
        if any_changes:
            self.encoded_tokens = new_encoded_tokens
            return True
        return False
    
    def build_vocabulary(self):
        """
        Build vocabulary using BPE algorithm on UTF-8 encodings:
        1. Find most frequent pair of UTF-8 codes
        2. Replace pair with a new token ID
        3. Repeat until desired vocabulary size is reached
        """
        print("\nStarting BPE algorithm on UTF-8 encodings...")
        
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Count current tokens for progress tracking
            tokens_before = sum(len(t) for t in self.encoded_tokens)
            
            # Find most common pair
            pairs = self.get_pairs()
            if not pairs:
                break
            
            most_common = pairs.most_common(1)[0]
            pair = most_common[0]
            freq = most_common[1]
            new_id = len(self.vocab) + self.initial_vocab_size
            
            # Merge pair if possible
            if self.merge_pair(pair, pair, new_id):
                self.vocab.add(new_id)
                tokens_after = sum(len(t) for t in self.encoded_tokens)
                print(f"Iteration {iteration} | Merged {pair} | {freq} times | New ID: {new_id} | Tokens: {tokens_before} → {tokens_after}")
            
            iteration += 1
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize input text using the built vocabulary"""
        tokens = []
        current_pos = 0
        
        while current_pos < len(text):
            longest_token = None
            longest_length = 0
            
            # Find the longest matching token
            for token in self.vocab:
                if text[current_pos:].startswith(token) and len(token) > longest_length:
                    longest_token = token
                    longest_length = len(token)
            
            if longest_token:
                tokens.append(self.token_to_id[longest_token])
                current_pos += longest_length
            else:
                tokens.append(self.token_to_id['<unk>'])
                current_pos += 1
        
        return tokens
    
    def get_utf8_encoding(self, text: str) -> List[int]:
        return [ord(char) for char in text]
    
    def get_compression_ratio(self) -> float:
        original_length = len(''.join(self.text.split()))
        compressed_length = len(self.vocab)
        return original_length / compressed_length if compressed_length > 0 else 0.0 
    
    def count_real_tokens(self, text: str) -> int:
        """Count actual tokens in text, including merged tokens (_X_)"""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == '_':
                # Handle merged token
                end = text.find('_', i + 1)
                if end != -1:
                    tokens.append(text[i:end+1])
                    i = end + 1
                    continue
            # Handle regular character
            if text[i] != ' ':
                tokens.append(text[i])
            i += 1
        return len(tokens)
    
    def get_stats(self):
        """
        Get tokenization statistics including:
        - Initial/final token counts
        - Initial/final vocabulary sizes
        - Compression ratio
        """
        final_tokens = sum(len(t) for t in self.encoded_tokens)
        return {
            "initial_tokens": self.initial_tokens_length,
            "initial_vocab": self.initial_vocab_size,
            "final_tokens": final_tokens,
            "final_vocab": len(self.vocab) + self.initial_vocab_size,
            "compression_ratio": self.initial_tokens_length / final_tokens if final_tokens > 0 else 0
        } 
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from tabulate import tabulate

class HindiTokenizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vocab_size = 1000
        self.special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>']
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0
        
        try:
            self.text = self.load_and_clean_text()
            self.vocab = set()
            self.initialize_vocab()
            self.build_vocabulary()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
    
    def load_and_clean_text(self) -> str:
        """Load and clean the text"""
        # Read file in chunks to handle large files
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
        text = re.sub(r'[०-९0-9]+', ' <num> ', text)
        text = re.sub(r'[A-Za-z]+', ' <eng> ', text)
        text = re.sub(r'[!@#$%^&*(),.?":{}|<>]', ' ', text)
        text = re.sub(r'[\u0964\u0965]', ' ', text)
        text = re.sub(r'[^\u0900-\u097F\s<>a-z]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<num>\s*<num>', '<num>', text)
        text = re.sub(r'<eng>\s*<eng>', '<eng>', text)
        
        print("Text cleaning complete!")
        return text.strip()
    
    def add_token(self, token: str):
        """Add a token to vocabulary with unique ID"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
            self.vocab.add(token)
    
    def initialize_vocab(self):
        """Initialize vocabulary with special tokens and individual characters"""
        # Add special tokens first
        for token in self.special_tokens:
            self.add_token(token)
        
        # Add individual characters
        for char in self.text:
            if char.strip():
                self.add_token(char)
    
    def get_pairs(self) -> Counter:
        """Get all adjacent pairs with their frequencies"""
        words = self.text.split()
        pairs = Counter()
        
        for word in words:
            # Convert word into tokens, skipping over ID tokens
            tokens = []
            i = 0
            while i < len(word):
                # Skip ID tokens (e.g., _72_)
                if word[i] == '_' and i + 1 < len(word):
                    end = word.find('_', i + 1)
                    if end != -1:
                        tokens.append(word[i:end+1])
                        i = end + 1
                        continue
                tokens.append(word[i])
                i += 1
            
            # Find pairs between tokens
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] += 1
        
        return pairs
    
    def merge_pair(self, pair: Tuple[str, str], merged_token: str, token_id: int) -> bool:
        """Replace pair with a unique ID token everywhere in text"""
        words = self.text.split()
        new_words = []
        any_changes = False
        
        # Create unique ID token
        id_token = f"_{token_id}_"
        
        for word in words:
            # Replace the pair in each word
            new_word = word
            pair_str = f"{pair[0]}{pair[1]}"
            while pair_str in new_word:  # Keep replacing until no more occurrences
                new_word = new_word.replace(pair_str, id_token)
                any_changes = True
            new_words.append(new_word)
        
        if any_changes:
            self.text = ' '.join(new_words)
            return True
        return False
    
    def build_vocabulary(self):
        """Build vocabulary using BPE"""
        # 1. Print initial stats
        initial_tokens = self.count_real_tokens(self.text)
        initial_vocab = len(self.vocab)
        print(f"1. Initial Length of tokens: {initial_tokens}")
        print(f"2. Initial vocabulary size: {initial_vocab}")
        print("\nStarting BPE algorithm...")
        
        iteration = 0
        while len(self.vocab) < self.vocab_size:
            # Get current token count
            tokens_before = self.count_real_tokens(self.text)
            
            # Find most common pair
            pairs = self.get_pairs()
            if not pairs:
                break
            
            most_common = pairs.most_common(1)[0]
            pair = most_common[0]
            freq = most_common[1]
            merged_token = ''.join(pair)
            new_id = len(self.vocab)
            
            # Try to merge
            if merged_token not in self.vocab:
                if self.merge_pair(pair, merged_token, new_id):
                    self.add_token(merged_token)
                    tokens_after = self.count_real_tokens(self.text)
                    print(f"Started | Iteration {iteration} | {merged_token} | {freq} times | {new_id} | {tokens_before} | {tokens_after}")
                
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
import argparse
import os
import random
import gzip
from tqdm import tqdm

def split_jsonl(input_file, output_dir, train_ratio=0.8, seed=42):
    """
    Split a raw JSONL file (supports .gz) into train and val sets based on line count.
    Maintains temporal order (does not shuffle).
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Select opener
    open_func = gzip.open if input_file.endswith(".gz") else open
    
    # 1. Count total lines
    print(f"Counting lines in {input_file}...")
    total_lines = 0
    with open_func(input_file, 'rb') as f:
        for _ in f:
            total_lines += 1
            
    print(f"Total lines: {total_lines}")
    
    # 2. Calculate split point
    split_idx = int(total_lines * train_ratio)
    print(f"Splitting at line {split_idx} (Train: {train_ratio*100}%, Val: {(1-train_ratio)*100}%)")
    
    train_file = os.path.join(output_dir, "train.jsonl.gz")
    val_file = os.path.join(output_dir, "val.jsonl.gz")
    
    # 3. Write files (Output as GZIP to save space)
    print("Writing files (compressed)...")
    
    # Input is already opened with correct opener (gzip or plain)
    # Output must be gzip
    with open_func(input_file, 'rt', encoding='utf-8') as fin, \
         gzip.open(train_file, 'wt', encoding='utf-8') as f_train, \
         gzip.open(val_file, 'wt', encoding='utf-8') as f_val:
        
        for i, line in enumerate(tqdm(fin, total=total_lines)):
            if i < split_idx:
                f_train.write(line)
            else:
                f_val.write(line)
                
    print(f"Done!")
    print(f"Train set: {train_file} ({split_idx} lines)")
    print(f"Val set:   {val_file} ({total_lines - split_idx} lines)")

def main():
    parser = argparse.ArgumentParser(description="Split raw JSONL dataset into Train and Val sets.")
    parser.add_argument("input_file", help="Path to the source JSONL file (e.g., normal.jsonl)")
    parser.add_argument("--output_dir", default="data/raw", help="Directory to save train.jsonl and val.jsonl")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of training data (default: 0.8)")
    
    args = parser.parse_args()
    
    split_jsonl(args.input_file, args.output_dir, args.ratio)

if __name__ == "__main__":
    main()

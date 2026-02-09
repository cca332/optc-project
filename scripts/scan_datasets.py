import glob
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from optc_uras.data.ecar import EcarJsonlReader

def scan_file(file_path):
    print(f"Scanning {os.path.basename(file_path)}...")
    reader = EcarJsonlReader(file_path)
    
    start_dt = None
    end_dt = None
    
    # Key: (host, window_idx) -> count
    # window_idx = timestamp_sec // (15 * 60)
    window_counts = defaultdict(int)
    
    total_events = 0
    
    for event in tqdm(reader, desc="Events", unit="evt"):
        total_events += 1
        dt = event["_dt"]
        host = event.get("hostname")
        
        if not start_dt or dt < start_dt:
            start_dt = dt
        if not end_dt or dt > end_dt:
            end_dt = dt
            
        if host:
            ts = dt.timestamp()
            w_idx = int(ts // 900) # 15 min * 60 sec = 900
            window_counts[(host, w_idx)] += 1
            
    if not window_counts:
        print("  No valid events found.")
        return

    counts = list(window_counts.values())
    
    print(f"  Time Range: {start_dt} to {end_dt}")
    print(f"  Total Events: {total_events}")
    print(f"  Total Samples (Host-15min windows): {len(counts)}")
    print(f"  Unique Hosts: {len(set(k[0] for k in window_counts.keys()))}")
    
    print("  Events per Sample Stats:")
    print(f"    Min: {np.min(counts)}")
    print(f"    Max: {np.max(counts)}")
    print(f"    Mean: {np.mean(counts):.2f}")
    print(f"    Median: {np.median(counts)}")
    print(f"    P90: {np.percentile(counts, 90)}")
    print(f"    P99: {np.percentile(counts, 99)}")
    print("-" * 40)

def main():
    base_dir = "/root/optc-project/data/raw"
    files = glob.glob(os.path.join(base_dir, "*.json.gz"))
    
    # Sort for consistent order
    files.sort()
    
    target_files = []
    test_files = []
    
    for f in files:
        if "25day" in f:
            test_files.append(f)
        else:
            target_files.append(f)
            
    print(f"Found {len(target_files)} training/val candidate files.")
    print(f"Found {len(test_files)} test files (skipped).")
    print("=" * 60)
    
    for f in target_files:
        scan_file(f)

if __name__ == "__main__":
    main()
import glob
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
# Ensure dependencies are available
try:
    from optc_uras.data.ecar import EcarJsonlReader
except ImportError:
    print("Error: Could not import EcarJsonlReader. Make sure you represent the directory structure correctly.")
    sys.exit(1)

def scan_file(file_path):
    print(f"\nScanning {os.path.basename(file_path)}...")
    try:
        reader = EcarJsonlReader(file_path)
    except Exception as e:
        print(f"  Error initializing reader: {e}")
        return

    start_dt = None
    end_dt = None
    
    # Key: (host, window_idx) -> count
    # window_idx = timestamp_sec // (15 * 60)
    window_counts = defaultdict(int)
    
    total_events = 0
    
    # Use tqdm for progress bar
    try:
        iterator = tqdm(reader, desc="  Events", unit="evt", mininterval=1.0)
    except Exception:
        iterator = reader

    for event in iterator:
        total_events += 1
        dt = event.get("_dt")
        if not dt:
            continue

        host = event.get("hostname")
        
        if not start_dt or dt < start_dt:
            start_dt = dt
        if not end_dt or dt > end_dt:
            end_dt = dt
            
        if host:
            ts = dt.timestamp()
            w_idx = int(ts // 900) # 15 min * 60 sec = 900
            window_counts[(host, w_idx)] += 1
            
    if not window_counts:
        print("  No valid events found.")
        return

    counts = list(window_counts.values())
    
    print(f"  Time Range: {start_dt} to {end_dt}")
    print(f"  Total Raw Events: {total_events}")
    print(f"  Total Samples (Host-15min windows): {len(counts)}")
    print(f"  Unique Hosts: {len(set(k[0] for k in window_counts.keys()))}")
    
    if len(counts) > 0:
        print("  Events per Sample Stats (for Sampling Strategy):")
        print(f"    Min: {np.min(counts)}")
        print(f"    Max: {np.max(counts)}")
        print(f"    Mean: {np.mean(counts):.2f}")
        print(f"    Median: {np.median(counts)}")
        print(f"    P25: {np.percentile(counts, 25)}")
        print(f"    P75: {np.percentile(counts, 75)}")
        print(f"    P90: {np.percentile(counts, 90)}")
    print("-" * 40)

def main():
    base_dir = "/root/optc-project/data/raw"
    # Find all .json.gz files
    files = glob.glob(os.path.join(base_dir, "*.json.gz"))
    
    if not files:
        print(f"No .json.gz files found in {base_dir}")
        return

    files.sort()
    
    target_files = []
    test_files = []
    
    for f in files:
        if "25day" in f:
            test_files.append(f)
        else:
            target_files.append(f)
            
    print(f"Found {len(target_files)} training/val candidate files.")
    print(f"Found {len(test_files)} test files (excluded from scan).")
    print("=" * 60)
    
    for f in target_files:
        scan_file(f)

if __name__ == "__main__":
    main()
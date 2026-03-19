import os
import sys
import argparse
import yaml
import pickle
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from optc_uras.data.ecar import EcarJsonlReader, WindowAggregator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_file_chunk(file_path: str, output_dir: str, prefix: str, cfg: dict, chunk_id: int):
    """Process a single chunk of JSONL file"""
    # Create unique prefix for this process to avoid file collision
    process_prefix = f"{prefix}_p{chunk_id:02d}"
    
    # Reuse existing process_file logic but with modified prefix
    process_file(file_path, output_dir, process_prefix, cfg)

def process_file(file_path: str, output_dir: str, prefix: str, cfg: dict):
    """Process a single JSONL file and save chunks"""
    
    # 1. Config Extraction
    time_range = None
    if cfg.get("time_range"):
        tr = cfg["time_range"]
        if tr.get("start") and tr.get("end"):
            time_range = (tr["start"], tr["end"])
            
    host_filter = cfg.get("host_filter")
    window_minutes = cfg.get("window_minutes", 15)
    max_events = cfg.get("max_events_per_window", 20000)
    view_mapping = cfg.get("view_mapping")   # 3. Pipeline Components
    reader = EcarJsonlReader(file_path, time_range=time_range, host_filter=host_filter)
    # [MODIFIED] Use Standard WindowAggregator (No Attack Prior)
    augment = cfg.get("augment", False)
    aggregator = WindowAggregator(window_minutes=window_minutes, 
                                  view_mapping=view_mapping, 
                                  max_events_per_window=max_events,
                                  augment=augment)
    
    # 3. Processing Loop
    chunk_size = cfg.get("chunk_size", 100)  # [OPTIMIZED] Read from config, default to 100
    chunk_buffer = []
    chunk_idx = 0
    total_samples = 0
    
    logger.info(f"Starting processing: {file_path}")
    # Only show config log for main process or if verbose
    # logger.info(f"Config: Window={window_minutes}m, MaxEvents={max_events}, Range={time_range}")
    
    iterator = aggregator.process(reader)
    
    # Reduce tqdm clutter in multiprocessing
    disable_tqdm = False
    if "p" in prefix and int(prefix.split("_p")[-1].split("_")[0]) > 0:
        disable_tqdm = True

    for sample in tqdm(iterator, desc=f"Processing {prefix}", disable=disable_tqdm):
        chunk_buffer.append(sample)
        if len(chunk_buffer) >= chunk_size:
            _save_chunk(chunk_buffer, output_dir, prefix, chunk_idx)
            total_samples += len(chunk_buffer)
            
            # Aggressive memory cleanup
            del chunk_buffer
            chunk_buffer = []
            chunk_idx += 1
            import gc
            gc.collect()
            
    # Save remaining
    if chunk_buffer:
        _save_chunk(chunk_buffer, output_dir, prefix, chunk_idx)
        total_samples += len(chunk_buffer)
        
    logger.info(f"Finished {prefix}. Total samples: {total_samples}")

def _save_chunk(data, output_dir, prefix, idx):
    filename = f"{prefix}_part{idx:03d}.pkl"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def split_and_process(file_path: str, output_dir: str, prefix: str, cfg: dict, num_workers: int = 20):
    """Split large file and process in parallel"""
    import subprocess
    import shutil
    
    temp_split_dir = os.path.join(output_dir, "split_temp")
    if not os.path.exists(temp_split_dir):
        os.makedirs(temp_split_dir)
    
    logger.info(f"Splitting {file_path} into chunks for parallel processing...")
    
    # Calculate lines per chunk (approximate)
    try:
        # [MODIFIED] Handle GZIP files: Decompress stream first then split
        output_prefix = os.path.join(temp_split_dir, "chunk_")
        
        # Determine chunk size strategy
        # For parallelism, we want roughly 'num_workers' chunks.
        # But file size based split is easier with 'split -C'
        # Let's target ~200MB compressed chunks to be safe on parallel RAM usage
        chunk_size = "200M"

        if file_path.endswith(".gz"):
            # [MODIFIED] Use --filter to compress chunks on the fly to save disk space!
            # Note: $FILE is provided by split. We append .json.gz
            cmd = f"zcat {file_path} | split -C {chunk_size} -d --filter='gzip > $FILE.json.gz' - {output_prefix}"
            logger.info(f"Executing: {cmd}")
            subprocess.check_call(cmd, shell=True)
        else:
            # Standard split for plain text
            # Also compress plain text chunks to save space!
            cmd = f"cat {file_path} | split -C {chunk_size} -d --filter='gzip > $FILE.json.gz' - {output_prefix}"
            logger.info(f"Executing: {cmd}")
            subprocess.check_call(cmd, shell=True)
            
    except Exception as e:
        logger.error(f"Failed to split file: {e}")
        return

    split_files = sorted([os.path.join(temp_split_dir, f) for f in os.listdir(temp_split_dir) if f.startswith("chunk_")])
    logger.info(f"Generated {len(split_files)} chunks. Starting parallel pool with {num_workers} workers.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, split_file in enumerate(split_files):
            futures.append(executor.submit(process_file_chunk, split_file, output_dir, prefix, cfg, i))
        
        for future in futures:
            future.result()  # Wait for all and propagate exceptions
            
    # Cleanup split files
    logger.info("Parallel processing complete. Cleaning up split files...")
    shutil.rmtree(temp_split_dir)

def process_shared_split(source_path: str, output_dir: str, train_prefix: str, val_prefix: str, train_cfg: dict, val_cfg: dict, split_ratio: float = 0.8, num_workers: int = 20):
    """
    Split source file ONCE and distribute chunks to Train/Val processing.
    This saves disk space by avoiding intermediate full train/val files.
    """
    import shutil
    import subprocess
    
    temp_split_dir = os.path.join(os.path.dirname(output_dir), "split_raw_shared")
    if os.path.exists(temp_split_dir):
        shutil.rmtree(temp_split_dir)
    os.makedirs(temp_split_dir)
    
    logger.info(f"Splitting shared source {source_path} for Train/Val processing...")
    
    try:
        output_prefix = os.path.join(temp_split_dir, "chunk_")
        chunk_size = "200M" # Compressed chunk size
        
        if source_path.endswith(".gz"):
            cmd = f"zcat {source_path} | split -C {chunk_size} -d --filter='gzip > $FILE.json.gz' - {output_prefix}"
        else:
            cmd = f"cat {source_path} | split -C {chunk_size} -d --filter='gzip > $FILE.json.gz' - {output_prefix}"
            
        logger.info(f"Executing split: {cmd}")
        subprocess.check_call(cmd, shell=True)
        
    except Exception as e:
        logger.error(f"Failed to split file: {e}")
        return

    split_files = sorted([os.path.join(temp_split_dir, f) for f in os.listdir(temp_split_dir) if f.startswith("chunk_")])
    total_chunks = len(split_files)
    split_idx = int(total_chunks * split_ratio)
    
    train_chunks = split_files[:split_idx]
    val_chunks = split_files[split_idx:]
    
    logger.info(f"Generated {total_chunks} chunks. Train: {len(train_chunks)}, Val: {len(val_chunks)}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Submit Train tasks
        for i, f in enumerate(train_chunks):
            futures.append(executor.submit(process_file_chunk, f, output_dir, train_prefix, train_cfg, i))
            
        # Submit Val tasks
        for i, f in enumerate(val_chunks):
            futures.append(executor.submit(process_file_chunk, f, output_dir, val_prefix, val_cfg, i))
            
        for future in futures:
            future.result()

    logger.info("Shared processing complete. Cleaning up...")
    shutil.rmtree(temp_split_dir)

def main():
    parser = argparse.ArgumentParser(description="Preprocess OpTC eCAR Data")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--override_train_path", type=str, help="Override train_path in config")
    parser.add_argument("--override_train_prefix", type=str, help="Override train_prefix in config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config.get("data", {}).get("optc", {})
    if not data_cfg:
        logger.error("Config missing 'data.optc' section")
        return

    cache_dir = data_cfg.get("cache_dir", "./cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Process Train (Truncated)
    train_path = args.override_train_path if args.override_train_path else data_cfg.get("train_path")
    train_prefix = args.override_train_prefix if args.override_train_prefix else data_cfg.get("train_prefix", "train")
    
    val_path = data_cfg.get("val_path")
    val_prefix = data_cfg.get("val_prefix", "val")
    
    num_workers = min(20, multiprocessing.cpu_count())

    # [OPTIMIZATION] Check for Shared Split Mode
    # [DISABLED] Disabling shared split optimization to avoid time window fragmentation.
    # Parallel splitting by bytes/lines breaks window integrity.
    # if train_path and val_path and train_path == val_path and os.path.exists(train_path):
    #     logger.info(f"Detected shared source for Train/Val: {train_path}")
    #     logger.info("Using Shared Split Mode to save space and time.")
    #     
    #     train_cfg = data_cfg.copy()
    #     train_cfg["augment"] = True
    #     
    #     val_cfg = data_cfg.copy()
    #     val_cfg["max_events_per_window"] = None
    #     
    #     process_shared_split(train_path, cache_dir, train_prefix, val_prefix, train_cfg, val_cfg, split_ratio=0.8, num_workers=num_workers)
    #     
    #     # Skip individual processing
    #     train_path = None
    #     val_path = None

    # [MODIFIED] Process Train Set
    # Handle list of files (comma separated string)
    train_path_exists = False
    if train_path:
        if "," in train_path:
            # Assume it's a list of files, check if at least one exists or just proceed
            # Better to check split
            paths = [p.strip() for p in train_path.split(",")]
            if any(os.path.exists(p) for p in paths):
                train_path_exists = True
        elif os.path.exists(train_path):
            train_path_exists = True

    if train_path and train_path_exists:
        logger.info(f"Processing Train Set: {train_path}")
        
        # Create train config (Augmentation disabled in RawReader now, but kept in config for compatibility)
        train_cfg = data_cfg.copy()
        # train_cfg["augment"] = True # Disabled in RawReader logic
        
        logger.info("Using sequential processing (no split) to maintain window integrity.")
        process_file(train_path, cache_dir, train_prefix, train_cfg)
        
    elif train_path:
        logger.warning(f"Train path not found or invalid: {train_path}")

    # [ADDED] Process Validation (Full/Complete - No Truncation)
    val_path_exists = False
    if val_path:
        if "," in val_path:
            paths = [p.strip() for p in val_path.split(",")]
            if any(os.path.exists(p) for p in paths):
                val_path_exists = True
        elif os.path.exists(val_path):
            val_path_exists = True

    if val_path and val_path_exists:
        logger.info(f"Processing Validation Set (Full - No Truncation): {val_path}")
        # Create a config copy with infinite max_events
        val_cfg = data_cfg.copy()
        # val_cfg["max_events_per_window"] = None  # [FIX] Avoid OOM by using config limit
        
        # [CRITICAL FIX] Force sequential for Val
        process_file(val_path, cache_dir, val_prefix, val_cfg)

    elif val_path:
        logger.warning(f"Validation path not found: {val_path}")

    # Process Test (Full/Complete - No Truncation)
    test_path = data_cfg.get("test_path")
    if test_path and os.path.exists(test_path):
        # [USER REQ] Use configurable prefix (default to "test")
        test_prefix = data_cfg.get("test_prefix", "test")
        logger.info(f"Processing Test Set (Full - No Truncation): {test_path}")
        # Create a config copy with infinite max_events
        test_cfg = data_cfg.copy()
        # test_cfg["max_events_per_window"] = None # [FIX] Avoid OOM by using config limit

        
        # [FIX] Force sequential processing for Test set to avoid window fragmentation
        # Splitting large files causes events for the same window to be split across workers,
        # resulting in multiple partial samples instead of one complete sample.
        logger.info("Forcing sequential processing for Test set to ensure correct aggregation...")
        process_file(test_path, cache_dir, test_prefix, test_cfg)
            
    elif test_path:
        logger.warning(f"Test path not found: {test_path}")

if __name__ == "__main__":
    main()

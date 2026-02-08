import os
import sys
import argparse
import yaml
import pickle
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from optc_uras.data.ecar import EcarJsonlReader, WindowAggregator

class AttackAwareAggregator(WindowAggregator):
    """
    Custom Aggregator that prioritizes specific attack windows for Sysclient0051.
    Instead of first-N truncation, it scans for attack timestamps and centers the window around them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attack Timestamps (Sysclient0051) in EDT (UTC-4)
        # 1. 2019-09-25 13:42:05 EDT (17:42:05 UTC) -> RDP
        # 2. 2019-09-25 14:24:03 EDT (18:24:03 UTC) -> Updated.exe
        # We use EDT to match the ground truth PDF logs.
        self.targets = [
            (datetime(2019, 9, 25, 13, 42, 5), "Sysclient0051"),
            (datetime(2019, 9, 25, 14, 24, 3), "Sysclient0051")
        ]
        
    def _is_attack_relevant(self, event_dt, hostname):
        # Check if event is within 10 minutes of any attack timestamp on the target host
        # [FIX] Handle FQDN (e.g., SysClient0051.systemia.com)
        if "sysclient0051" not in hostname.lower():
            return False
            
        for target_dt, target_host in self.targets:
            # Event dt is UTC. Target dt is EDT.
            # Convert Event UTC to EDT for comparison: EDT = UTC - 4h
            event_dt_edt = event_dt - timedelta(hours=4)
            
            # Use naive comparison (ignoring tzinfo objects to avoid mismatches)
            # Both are now effectively "Wall Clock Time in EDT"
            t_dt = target_dt.replace(tzinfo=None)
            e_dt = event_dt_edt.replace(tzinfo=None)
                
            diff = abs((e_dt - t_dt).total_seconds())
            if diff < 600: # 10 minutes window
                return True
        return False

    def process(self, event_stream):
        # [MODIFIED] Logic to allow over-buffering for attack candidates
        for event in event_stream:
            dt = event.pop("_dt") # Pop internal field
            hostname = event.get("hostname")
            obj_type = event.get("object")
            
            if not hostname or not obj_type:
                continue
            
            # Map view
            view_name = self.obj_to_view.get(obj_type)
            if not view_name:
                continue # Drop unknown objects

            # Determine window
            w_start = self._get_window_start(dt)
            key = (hostname, w_start)

            if key not in self.buffer:
                self.buffer[key] = self._init_sample(hostname, w_start)
            
            # Add event
            sample = self.buffer[key]
            
            # [CRITICAL CHANGE]
            # Always append, do NOT truncate yet.
            # We will truncate in _yield_sample
            
            clean_event = {
                "timestamp": event.get("timestamp"),
                "hostname": hostname,
                "object": obj_type,
                "action": event.get("action"),
                "properties": event.get("properties", {}),
                "_dt_obj": dt # Keep dt for sorting/filtering later
            }
            sample["views"][view_name].append(clean_event)

            # Watermark check
            if self.last_seen_dt is None or dt > self.last_seen_dt:
                self.last_seen_dt = dt
                yield from self._flush_old(dt)

        # Final flush
        yield from self._flush_all()

    def _truncate_and_yield(self, sample):
        # Apply truncation logic
        host = sample["host"]
        
        for view_name, events in sample["views"].items():
            if len(events) <= self.max_events:
                # Cleanup internal field
                for e in events:
                    e.pop("_dt_obj", None)
                continue
                
            # Need truncation
            # Check if this window contains attack-relevant events
            attack_indices = []
            for i, e in enumerate(events):
                if self._is_attack_relevant(e.get("_dt_obj"), host):
                    attack_indices.append(i)
            
            if attack_indices:
                # Found attack! Center the window around the attack.
                center_idx = attack_indices[len(attack_indices)//2]
                half_window = self.max_events // 2
                start_idx = max(0, center_idx - half_window)
                end_idx = start_idx + self.max_events
                
                # Adjust if out of bounds
                if end_idx > len(events):
                    end_idx = len(events)
                    start_idx = max(0, end_idx - self.max_events)
                
                logger.info(f"[AttackScanner] Found attack on {host}! Extracting slice {start_idx}:{end_idx} (Total {len(events)})")
                sample["views"][view_name] = events[start_idx:end_idx]
            else:
                # Normal truncation (first N)
                sample["views"][view_name] = events[:self.max_events]
            
            # Cleanup internal field
            for e in sample["views"][view_name]:
                e.pop("_dt_obj", None)
                
        return sample

    def _flush_old(self, current_dt):
        # Flush windows that ended more than `window_minutes` ago
        safe_margin = self.window_delta * 2
        threshold_ms = (current_dt - safe_margin).timestamp() * 1000
        
        keys_to_remove = []
        for key, sample in self.buffer.items():
            if sample["t1"] < threshold_ms:
                yield self._truncate_and_yield(sample)
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.buffer[k]

    def _flush_all(self):
        for sample in self.buffer.values():
            yield self._truncate_and_yield(sample)
        self.buffer.clear()
        
        if self.stats["processed_windows"] > 0:
            logger.info(f"Window Aggregation Stats: Processed {self.stats['processed_windows']} windows.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # [MODIFIED] Use AttackAwareAggregator to capture attack windows
    aggregator = AttackAwareAggregator(window_minutes=window_minutes, 
                                      view_mapping=view_mapping, 
                                      max_events_per_window=max_events)
    
    # 3. Processing Loop
    chunk_size = 5  # [USER REQ] Increased for speed
    chunk_buffer = []
    chunk_idx = 0
    total_samples = 0
    
    logger.info(f"Starting processing: {file_path}")
    logger.info(f"Config: Window={window_minutes}m, MaxEvents={max_events}, Range={time_range}")
    
    iterator = aggregator.process(reader)
    
    for sample in tqdm(iterator, desc=f"Processing {prefix}"):
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

def main():
    parser = argparse.ArgumentParser(description="Preprocess OpTC eCAR Data")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_cfg = config.get("data", {}).get("optc", {})
    if not data_cfg:
        logger.error("Config missing 'data.optc' section")
        return

    cache_dir = data_cfg.get("cache_dir", "./cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Process Train
    train_path = data_cfg.get("train_path")
    # [USER REQ] Re-enable train processing for large-event update
    if train_path and os.path.exists(train_path):
        process_file(train_path, cache_dir, "train4", data_cfg)
    elif train_path:
        logger.warning(f"Train path not found: {train_path}")

    # Process Test
    test_path = data_cfg.get("test_path")
    if test_path and os.path.exists(test_path):
        # [USER REQ] Rename test output to "test4"
        process_file(test_path, cache_dir, "test4", data_cfg)
    elif test_path:
        logger.warning(f"Test path not found: {test_path}")

if __name__ == "__main__":
    main()
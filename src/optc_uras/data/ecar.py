"""
OpTC eCAR data reader and window aggregator.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EcarJsonlReader:
    """流式读取 OpTC eCAR JSONL 数据"""

    def __init__(self, file_path: str, 
                 time_range: Optional[Tuple[str, str]] = None,
                 host_filter: Optional[List[str]] = None):
        self.file_path = file_path
        
        # Parse time range (ISO8601 strings)
        self.start_dt: Optional[datetime] = None
        self.end_dt: Optional[datetime] = None
        if time_range:
            try:
                self.start_dt = datetime.fromisoformat(time_range[0])
                self.end_dt = datetime.fromisoformat(time_range[1])
            except ValueError as e:
                logger.error(f"Invalid time format in time_range: {e}")
                raise

        self.host_set: Optional[Set[str]] = set(host_filter) if host_filter else None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        try:
            import gzip
            open_func = gzip.open if self.file_path.endswith(".gz") else open
            with open_func(self.file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # 1. Host Filter
                    hostname = event.get("hostname")
                    if self.host_set and hostname not in self.host_set:
                        continue

                    # 2. Time Filter
                    ts_str = event.get("timestamp")
                    if not ts_str:
                        continue
                    
                    try:
                        dt = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue

                    if self.start_dt and dt < self.start_dt:
                        continue
                    if self.end_dt and dt > self.end_dt:
                        continue

                    # Keep essential fields + parsed datetime
                    event["_dt"] = dt  # Internal use
                    yield event

        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise


class WindowAggregator:
    """将事件流聚合为 (host, 15min) 样本"""

    DEFAULT_VIEW_MAPPING = {
        "network": {"FLOW"},
        "file": {"FILE", "REGISTRY"},
        "process": {"PROCESS", "THREAD", "MODULE", "SHELL", "TASK", "USER_SESSION"}
    }

    def __init__(self, window_minutes: int = 15,
                 view_mapping: Optional[Dict[str, List[str]]] = None,
                 max_events_per_window: int = 20000,
                 augment: bool = False):
        self.window_delta = timedelta(minutes=window_minutes)
        self.max_events = max_events_per_window
        self.augment = augment # [ADDED] Toggle augmentation logic
        self.event_counter = 0  # For controlling flush frequency
        
        # Statistics for dropped events
        self.stats = {
            "processed_windows": 0,
            "dropped_events": defaultdict(int), # view_name -> count
            "truncated_windows": defaultdict(int) # view_name -> count
        }
        
        # Build optimized mapping: object_type -> view_name
        self.obj_to_view: Dict[str, str] = {}
        mapping = view_mapping if view_mapping else self.DEFAULT_VIEW_MAPPING
        for view_name, obj_types in mapping.items():
            for obj in obj_types:
                self.obj_to_view[obj] = view_name

        # Buffer: (host, window_start_ts) -> SampleDict
        self.buffer: Dict[Tuple[str, int], Dict[str, Any]] = {}
        
        # Watermark for cleanup (assuming roughly sorted input)
        self.last_seen_dt: Optional[datetime] = None

    def _get_window_start(self, dt: datetime) -> int:
        # Align to window boundary (unix timestamp in ms)
        # Using simple epoch division
        ts = dt.timestamp()
        window_sec = self.window_delta.total_seconds()
        start_ts = int(ts // window_sec) * int(window_sec)
        return int(start_ts * 1000) # ms

    def _init_sample(self, hostname: str, start_ms: int) -> Dict[str, Any]:
        end_ms = start_ms + int(self.window_delta.total_seconds() * 1000)
        return {
            "host": hostname,
            "t0": start_ms,
            "t1": end_ms,
            "views": {v: [] for v in set(self.obj_to_view.values())},
            "label": 0,  # Default benign
            "_truncated_views": set()  # Track which views have been truncated
        }

    def process(self, event_stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
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
            
            # Add event if not full (Global Limit Check)
            sample = self.buffer[key]
            current_total = sum(len(v) for v in sample["views"].values())
            
            # [MODIFIED] Check max_events (None/inf means infinite)
            # If augment is True, we collect ALL events first (no truncation), then sample later.
            is_full = False
            if not self.augment and self.max_events is not None and self.max_events != float('inf'):
                if current_total >= self.max_events:
                    is_full = True

            if not is_full:
                # Calculate relative timestamp (seconds)
                event_ts = dt.timestamp()
                window_start_ts = w_start / 1000.0
                rel_ts = max(0.0, event_ts - window_start_ts) # Ensure non-negative

                # Essential fields for training (Aligned with Doc/Step1)
                clean_event = {
                    # Required fields
                    "ts": rel_ts,
                    "type": event.get("action", "unknown"),
                    "op": event.get("action", "unknown"),
                    "obj": obj_type,
                    "valid": 1,
                    
                    # Original fields
                    "timestamp": event.get("timestamp"),
                    "hostname": hostname,
                    "object": obj_type,
                    "action": event.get("action"),
                    "properties": event.get("properties", {})
                }

                # [ADDED] Extract specific fields required by Step1 Quality Metrics
                # Try to find them in properties first, then top-level
                props = event.get("properties", {})
                for field in ["image_path", "command_line", "file_path", "dest_ip", "dest_port"]:
                    if field in props:
                        clean_event[field] = props[field]
                    elif field in event:
                        clean_event[field] = event[field]

                sample["views"][view_name].append(clean_event)
            else:
                # Track dropped stats
                self.stats["dropped_events"][view_name] += 1
                
                # Count this window as truncated only once per view
                if view_name not in sample["_truncated_views"]:
                    self.stats["truncated_windows"][view_name] += 1
                    sample["_truncated_views"].add(view_name)

            # Watermark check for flush (simple strategy: flush windows older than 2 windows ago)
            # This assumes data is mostly sorted.
            if self.last_seen_dt is None or dt > self.last_seen_dt:
                self.last_seen_dt = dt
                yield from self._flush_old(dt)

        # Final flush
        yield from self._flush_all()

    def _apply_augmentation(self, raw_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply augmentation strategies based on total event count"""
        import random
        import copy
        
        # 1. Calculate total events
        total_events = sum(len(v) for v in raw_sample["views"].values())
        
        # 2. Determine number of augmentations
        # Rule: <10k: 1x, 10k-30k: 3x, 30k-60k: 5x, >60k: 10x
        if total_events < 10000:
            num_augmentations = 1
        elif total_events < 30000:
            num_augmentations = 3
        elif total_events < 60000:
            num_augmentations = 5
        else:
            num_augmentations = 10
            
        # Target size for each sample (default to max_events or 8000 if not set)
        target_size = self.max_events if (self.max_events and self.max_events != float('inf')) else 8000
        
        augmented_samples = []
        
        # Flatten all events for global sampling
        all_events = []
        for v_name, events in raw_sample["views"].items():
            for evt in events:
                all_events.append((v_name, evt))
                
        for i in range(num_augmentations):
            # Deep copy base structure
            aug_sample = {
                "host": raw_sample["host"],
                "t0": raw_sample["t0"],
                "t1": raw_sample["t1"],
                "label": raw_sample["label"],
                "aug_id": i, # Track augmentation ID
                "views": {v: [] for v in raw_sample["views"].keys()}
            }
            
            # 3. Global Random Sampling
            # If total events > target_size, sample randomly. Else keep all.
            if len(all_events) > target_size:
                sampled_pairs = random.sample(all_events, target_size)
            else:
                sampled_pairs = list(all_events)
            
            # 4. Re-distribute to views
            for v_name, evt in sampled_pairs:
                aug_sample["views"][v_name].append(evt)
            
            augmented_samples.append(aug_sample)
            
        return augmented_samples

    def _flush_old(self, current_dt: datetime) -> Iterator[Dict[str, Any]]:
        # Flush windows that ended more than `window_minutes` ago relative to current_dt
        # giving a safe margin for out-of-order events.
        safe_margin = self.window_delta * 2
        threshold_ms = (current_dt - safe_margin).timestamp() * 1000
        
        keys_to_remove = []
        for key, sample in self.buffer.items():
            if sample["t1"] < threshold_ms:
                # Remove internal tracking field before yielding
                sample.pop("_truncated_views", None)
                self.stats["processed_windows"] += 1
                
                if self.augment:
                    # Yield multiple augmented samples
                    for aug_s in self._apply_augmentation(sample):
                        yield aug_s
                else:
                    yield sample
                    
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.buffer[k]

    def _flush_all(self) -> Iterator[Dict[str, Any]]:
        for sample in self.buffer.values():
            sample.pop("_truncated_views", None)
            self.stats["processed_windows"] += 1
            
            if self.augment:
                for aug_s in self._apply_augmentation(sample):
                    yield aug_s
            else:
                yield sample
                
        self.buffer.clear()
        
        # Log final stats
        if self.stats["processed_windows"] > 0:
            logger.info(f"Window Aggregation Stats: Processed {self.stats['processed_windows']} windows.")
            for v, count in self.stats["dropped_events"].items():
                wins = self.stats["truncated_windows"][v]
                logger.warning(f"View '{v}': Dropped {count} events in {wins} truncated windows (Max={self.max_events}).")
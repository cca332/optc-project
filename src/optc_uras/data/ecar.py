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
            
            # Handle comma-separated files or single file
            files_to_read = []
            if "," in self.file_path:
                files_to_read = [p.strip() for p in self.file_path.split(",")]
            else:
                files_to_read = [self.file_path]
                
            for current_file in files_to_read:
                open_func = gzip.open if current_file.endswith(".gz") else open
                with open_func(current_file, "rt", encoding="utf-8") as f:
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
    """
    将事件流聚合为 (host, window) 样本。
    
    Update 2026-03:
    - Default window size reduced to 5 min.
    - View mapping upgraded to 6 fine-grained views.
    - Max events truncation removed.
    - Augmentation removed.
    - Semantic fields extraction added.
    """

    # Reverted to 3-view mapping but including HOST
    DEFAULT_VIEW_MAPPING = {
        "network": {"FLOW"},
        "file": {"FILE", "REGISTRY"},
        "process": {"PROCESS", "THREAD", "MODULE", "SHELL", "TASK", "USER_SESSION", "HOST"}
    }

    def __init__(self, window_minutes: int = 5,
                 view_mapping: Optional[Dict[str, List[str]]] = None,
                 max_events_per_window: Optional[int] = None, # Deprecated, defaults to infinite
                 augment: bool = False): # Deprecated, disabled
        self.window_delta = timedelta(minutes=window_minutes)
        # Force disable max_events and augment as per new requirements
        self.max_events = float('inf') 
        self.augment = False
        
        # Statistics
        self.stats = {
            "processed_windows": 0,
            "total_events": 0
        }
        
        # Build optimized mapping: object_type -> view_name
        self.obj_to_view: Dict[str, str] = {}
        mapping = view_mapping if view_mapping else self.DEFAULT_VIEW_MAPPING
        for view_name, obj_types in mapping.items():
            for obj in obj_types:
                self.obj_to_view[obj] = view_name

        # Buffer: (host, window_start_ts) -> SampleDict
        self.buffer: Dict[Tuple[str, int], Dict[str, Any]] = {}
        
        # Track last event ts for delta_t calculation: (host, window_start, view) -> last_ts
        self.last_ts_map: Dict[Tuple[str, int, str], float] = {}
        
        # Watermark for cleanup
        self.last_seen_dt: Optional[datetime] = None

    def _get_window_start(self, dt: datetime) -> int:
        # Align to window boundary (unix timestamp in ms)
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
        }
    
    def _normalize_path(self, path: str) -> str:
        if not path:
            return ""
        return path.lower().replace("/", "\\")

    def _make_semantic_obj(self, event: Dict[str, Any], obj_type: str) -> str:
        """Extract the main semantic key (Subject) based on object type."""
        props = event.get("properties", {})
        
        # Helper to get field from props or event root
        def get(key):
            return props.get(key) or event.get(key, "")

        if obj_type in ["PROCESS", "THREAD"]:
            # Priority: image_path > command_line > parent
            img = self._normalize_path(get("image_path"))
            cmd = get("command_line")
            if img:
                return img
            return cmd
            
        elif obj_type == "FILE":
            return self._normalize_path(get("file_path"))
            
        elif obj_type == "FLOW":
            # Semantic: dest_ip:dest_port/proto
            dip = get("dest_ip")
            dport = get("dest_port")
            proto = get("l4protocol")
            direction = get("direction")
            if dip and dport:
                return f"{dip}:{dport}/{proto}/{direction}"
            return f"{get('src_ip')}:{get('src_port')}->{dip}:{dport}"
            
        elif obj_type == "REGISTRY":
            k = self._normalize_path(get("key"))
            v = get("value")
            return f"{k}\\{v}"
            
        elif obj_type == "MODULE":
            # module_path > image_path
            mod = self._normalize_path(get("module_path"))
            if mod:
                return mod
            return self._normalize_path(get("image_path"))
            
        elif obj_type == "SHELL":
            # payload > command_line > image_path
            payload = get("payload")
            if payload:
                return payload[:200] # Truncate long payloads
            cmd = get("command_line")
            if cmd:
                return cmd
            return self._normalize_path(get("image_path"))
            
        elif obj_type == "TASK":
            return self._normalize_path(get("image_path"))
            
        elif obj_type == "USER_SESSION":
            # src_ip:src_port
            sip = get("src_ip")
            if sip:
                return f"{sip}:{get('src_port')}"
            return get("image_path")
            
        # Fallback
        return self._normalize_path(get("image_path")) or "unknown"

    def _compute_valid(self, event: Dict[str, Any], obj_type: str) -> int:
        """Check if essential fields exist."""
        props = event.get("properties", {})
        def has(key):
            return bool(props.get(key) or event.get(key))

        if obj_type == "FILE":
            return 1 if has("file_path") else 0
        if obj_type == "FLOW":
            return 1 if (has("dest_ip") or has("src_ip")) else 0
        if obj_type == "REGISTRY":
            return 1 if has("key") else 0
        if obj_type == "MODULE":
            return 1 if has("module_path") else 0
        if obj_type in ["PROCESS", "SHELL"]:
            return 1 if (has("image_path") or has("command_line") or has("payload")) else 0
            
        return 1

    def process(self, event_stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for event in event_stream:
            dt = event.pop("_dt") 
            hostname = event.get("hostname")
            obj_type = event.get("object")
            
            if not hostname or not obj_type:
                continue
            
            view_name = self.obj_to_view.get(obj_type)
            if not view_name:
                continue 

            w_start = self._get_window_start(dt)
            key = (hostname, w_start)

            if key not in self.buffer:
                self.buffer[key] = self._init_sample(hostname, w_start)
            
            sample = self.buffer[key]
            
            # Calculate timestamps
            event_ts = dt.timestamp()
            window_start_ts = w_start / 1000.0
            rel_ts = max(0.0, event_ts - window_start_ts)
            
            # Calculate delta_t
            last_ts_key = (hostname, w_start, view_name)
            last_ts = self.last_ts_map.get(last_ts_key, window_start_ts)
            delta_t = max(0.0, event_ts - last_ts)
            self.last_ts_map[last_ts_key] = event_ts

            # Semantic extraction
            semantic_obj = self._make_semantic_obj(event, obj_type)
            is_valid = self._compute_valid(event, obj_type)

            # Construct clean event
            clean_event = {
                # Core fields for Step1
                "ts": rel_ts,
                "delta_t": delta_t,
                "type": obj_type, # [CHANGED] type is now Original Object Type (e.g. "PROCESS")
                "op": event.get("action", "unknown"), # [CHANGED] op is Action
                "obj": semantic_obj, # [CHANGED] obj is Semantic Key
                "view": view_name,
                "fine_view": obj_type, # Fine granularity view
                "valid": is_valid,
                
                # Metadata
                "timestamp": event.get("timestamp"),
                "hostname": hostname,
                "object": obj_type,
                "action": event.get("action"),
            }

            # Flatten properties to top-level
            props = event.get("properties", {})
            # Fields to extract
            extract_fields = [
                "image_path", "command_line", "parent_image_path",
                "file_path", "new_path",
                "dest_ip", "dest_port", "src_ip", "src_port", "l4protocol", "direction",
                "key", "value", "data", "type", # reg type
                "module_path", "payload"
            ]
            
            for f in extract_fields:
                val = props.get(f) or event.get(f)
                if val:
                    clean_event[f] = str(val)

            sample["views"][view_name].append(clean_event)
            self.stats["total_events"] += 1

            # Watermark check
            if self.last_seen_dt is None or dt > self.last_seen_dt:
                self.last_seen_dt = dt
                yield from self._flush_old(dt)

        yield from self._flush_all()

    def _flush_old(self, current_dt: datetime) -> Iterator[Dict[str, Any]]:
        safe_margin = self.window_delta * 2
        threshold_ms = (current_dt - safe_margin).timestamp() * 1000
        
        keys_to_remove = []
        for key, sample in self.buffer.items():
            if sample["t1"] < threshold_ms:
                self.stats["processed_windows"] += 1
                yield sample
                keys_to_remove.append(key)
                
                # Cleanup last_ts_map for this window
                hostname, w_start = key
                for view in self.obj_to_view.values():
                     self.last_ts_map.pop((hostname, w_start, view), None)
        
        for k in keys_to_remove:
            del self.buffer[k]

    def _flush_all(self) -> Iterator[Dict[str, Any]]:
        for key, sample in self.buffer.items():
            self.stats["processed_windows"] += 1
            yield sample
            
        self.buffer.clear()
        self.last_ts_map.clear()
        
        if self.stats["processed_windows"] > 0:
            logger.info(f"Stats: {self.stats['processed_windows']} windows, {self.stats['total_events']} events.")
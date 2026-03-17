import json
from datetime import datetime
from dateutil import parser
from pathlib import Path

file_path = "data/raw/AIA-51-75.ecar-last.json"

total_records = 0
min_ts = None
max_ts = None

print("Scanning file...")

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts_str = event.get("timestamp")
        if not ts_str:
            continue

        ts = parser.parse(ts_str)

        if min_ts is None or ts < min_ts:
            min_ts = ts

        if max_ts is None or ts > max_ts:
            max_ts = ts

        total_records += 1


print("\n===== Statistics =====")
print(f"Total events: {total_records}")

print(f"Earliest timestamp: {min_ts}")
print(f"Latest timestamp:   {max_ts}")

time_span = max_ts - min_ts
minutes = time_span.total_seconds() / 60

print(f"Total time span: {time_span} ({minutes:.2f} minutes)")

# 5 minute window
window_minutes = 5
num_samples = int(minutes // window_minutes) + 1

print(f"\nIf using {window_minutes}-minute windows:")
print(f"Estimated samples: {num_samples}")
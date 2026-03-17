import os
import pickle
from pprint import pprint


def main():
    cache_dir = os.path.join("data", "cache")
    # Prefer train_part, fall back to test_part
    candidates = []
    for name in os.listdir(cache_dir):
        if name.startswith("train_part") and name.endswith(".pkl"):
            candidates.append(name)
    if not candidates:
        for name in os.listdir(cache_dir):
            if name.startswith("test_part") and name.endswith(".pkl"):
                candidates.append(name)
    if not candidates:
        print("No *_part*.pkl files found in", cache_dir)
        return

    fname = sorted(candidates)[0]
    path = os.path.join(cache_dir, fname)
    print("Using file:", path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("num_samples_in_file:", len(data))
    if not data:
        print("File is empty.")
        return

    sample = data[0]
    print("keys:", list(sample.keys()))
    print("host:", sample.get("host"))
    print("t0:", sample.get("t0"))
    print("t1:", sample.get("t1"))
    print("views sizes:", {k: len(v) for k, v in sample.get("views", {}).items()})

    slot_seconds = 30

    def summarize_view(view_name, max_events=5):
        evs = sample.get("views", {}).get(view_name, [])[:max_events]
        out = []
        for e in evs:
            ts = e.get("ts")
            slot = int(ts // slot_seconds) if ts is not None else None
            out.append(
                {
                    "timestamp": e.get("timestamp"),
                    "ts": ts,
                    "slot": slot,
                    "type": e.get("type"),
                    "op": e.get("op"),
                    "obj": e.get("obj"),
                }
            )
        return out

    summary = {
        "process": summarize_view("process"),
        "file": summarize_view("file"),
        "network": summarize_view("network"),
    }
    print("sample_events_by_view:")
    pprint(summary)


if __name__ == "__main__":
    main()


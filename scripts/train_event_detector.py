import os
import sys
import json
import pickle
import copy
import yaml
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from optc_uras.data.dataset import OpTCEcarDataset
from optc_uras.models.event_encoder import EventWindowEncoder
from optc_uras.models.event_detector import EventMaskedReconstructionDetector
from optc_uras.federated.dp import GradDPConfig, clip_by_l2_norm_, add_gaussian_noise_
from optc_uras.federated.server import FederatedServer, ServerConfig


def deep_update(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_merged_config(config_path, event_config_path=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    resolved_event_path = event_config_path or os.path.join(os.path.dirname(config_path), "event_method.yaml")
    if os.path.exists(resolved_event_path):
        with open(resolved_event_path, "r", encoding="utf-8") as f:
            event_cfg = yaml.safe_load(f) or {}
        config = deep_update(config, event_cfg)
    return config, resolved_event_path


class EventAugmentedDataset:
    def __init__(self, base_dataset, aux_map):
        self.base_dataset = base_dataset
        self.aux_map = aux_map or {}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = dict(self.base_dataset[idx])
        key = (sample.get("host", "unknown"), int(sample.get("t0", 0)), int(sample.get("label", 0)))
        aux = self.aux_map.get(key)
        if aux is not None:
            sample["_cached_stat_vecs"] = aux["stat_vecs"]
            sample["_cached_quality"] = aux["quality"]
        return sample

    def get_metadata(self, idx):
        if hasattr(self.base_dataset, "get_metadata"):
            meta = dict(self.base_dataset.get_metadata(idx))
        else:
            sample = self.base_dataset[idx]
            meta = {
                "host": sample.get("host", "unknown"),
                "t0": int(sample.get("t0", 0)),
                "label": int(sample.get("label", 0)),
            }
        if "label" not in meta:
            sample = self.base_dataset[idx]
            meta["label"] = int(sample.get("label", 0))
        return meta


class EventCachedDataset:
    def __init__(self, split_length, aux_map, event_cache_reader):
        self.split_length = int(split_length)
        self.aux_map = aux_map or {}
        self.event_cache_reader = event_cache_reader

    def __len__(self):
        return self.split_length

    def __getitem__(self, idx):
        event_cache = self.event_cache_reader[idx]
        meta = event_cache.get("metadata", {}) if event_cache is not None else {}
        sample = {
            "host": meta.get("host", "unknown"),
            "t0": int(meta.get("t0", 0)),
            "label": int(meta.get("label", 0)),
            "_event_cache": event_cache,
        }
        key = (sample.get("host", "unknown"), int(sample.get("t0", 0)), int(sample.get("label", 0)))
        aux = self.aux_map.get(key)
        if aux is not None:
            sample["_cached_stat_vecs"] = aux["stat_vecs"]
            sample["_cached_quality"] = aux["quality"]
        return sample

    def get_metadata(self, idx):
        meta = self.event_cache_reader.get_metadata(idx)
        return {
            "host": meta.get("host", "unknown"),
            "t0": int(meta.get("t0", 0)),
            "label": int(meta.get("label", 0)),
        }


def setup_environment(config_path, event_config_path=None):
    config, resolved_event_path = load_merged_config(config_path, event_config_path=event_config_path)
    optc_cfg = config["data"]["optc"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    event_cfg = config.get("event_method", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return optc_cfg, model_cfg, train_cfg, event_cfg, device, resolved_event_path


def load_cached_aux(cache_dir):
    cache_path = os.path.join(cache_dir, "optimized_data.pt")
    if not os.path.exists(cache_path):
        return {}
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = data.get("metadata", [])
    stat_vecs = data.get("stat_vecs")
    quality = data.get("quality")
    if stat_vecs is None or quality is None or len(metadata) == 0:
        return {}

    aux_map = {}
    for i, meta in enumerate(metadata):
        key = (meta.get("host", "unknown"), int(meta.get("t0", 0)), int(meta.get("label", 0)))
        aux_map[key] = {
            "stat_vecs": stat_vecs[i],
            "quality": quality[i],
        }
    return aux_map


class EventCacheReader:
    def __init__(self, cache_dir, split):
        self.index_path = os.path.join(cache_dir, f"event_index_cache_{split}.pkl")
        self.index_entries = []
        self._last_fpath = None
        self._last_chunk = None
        self.cache_is_dirty = False
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                self.index_entries = pickle.load(f)
            self._quick_validate()

    def _quick_validate(self):
        if not self.index_entries:
            return
        fpath, local_idx = self.index_entries[0]
        try:
            chunk = torch.load(fpath, map_location="cpu", weights_only=False)
            item = chunk[local_idx] if local_idx < len(chunk) else {}
        except Exception:
            self.cache_is_dirty = True
            return
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        has_event_payload = bool(item.get("raw_events")) or bool(item.get("event_meta"))
        self.cache_is_dirty = not (bool(meta) and meta.get("host") not in (None, "", "unknown") and int(meta.get("t0", 0)) > 0 and has_event_payload)

    def __len__(self):
        return len(self.index_entries)

    def __getitem__(self, idx):
        if idx >= len(self.index_entries):
            return None
        fpath, local_idx = self.index_entries[idx]
        if self._last_fpath != fpath or self._last_chunk is None:
            self._last_chunk = torch.load(fpath, map_location="cpu", weights_only=False)
            self._last_fpath = fpath
        item = self._last_chunk[local_idx]
        if "metadata" not in item:
            item["metadata"] = {}
        if "raw_events" not in item:
            item["raw_events"] = []
        return item

    def get_metadata(self, idx):
        item = self[idx]
        return item.get("metadata", {})


def batch_to_rows(batch_samples, scores, threshold, per_event_scores, event_meta):
    rows = []
    preds = (scores > threshold).astype(int)
    for i, sample in enumerate(batch_samples):
        top_view = "unknown"
        sample_meta = event_meta[i]
        if sample_meta:
            valid_len = min(len(sample_meta), per_event_scores.shape[1])
            top_idx = int(np.argmax(per_event_scores[i, :valid_len]))
            top_view = sample_meta[top_idx].get("view_name", "unknown")
        dt_utc = pd.to_datetime(sample.get("t0", 0), unit="ms")
        rows.append(
            {
                "timestamp": dt_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                "host": sample.get("host", "unknown"),
                "anomaly_score": float(scores[i]),
                "adaptive_threshold": float(threshold),
                "base_threshold": float(threshold),
                "is_anomaly": int(preds[i]),
                "top_attribution": top_view,
            }
        )
    return rows


def filter_prediction_rows(df):
    if df.empty:
        return df
    keep = ~(
        (df["host"].astype(str) == "unknown")
        & (df["timestamp"].astype(str) == "1970-01-01T00:00:00")
        & (df["anomaly_score"].astype(float) == 0.0)
    )
    removed = int((~keep).sum())
    if removed > 0:
        print(f"[EventDetector] Removed {removed} invalid placeholder rows before evaluation")
    return df.loc[keep].reset_index(drop=True)


def compute_scores(loader, encoder, detector, desc="Scoring"):
    all_scores = []
    all_labels = []
    all_rows = []
    encoder.eval()
    detector.eval()
    with torch.no_grad():
        for batch_samples in tqdm(loader, desc=desc):
            enc = encoder(batch_samples)
            res = detector.compute_event_scores(enc["event_embeddings"], enc["event_mask"], leave_one_out=True)
            scores = res["score"].cpu().numpy()
            per_event_scores = res["per_event_score"].cpu().numpy()
            threshold = float(detector.threshold.item())
            batch_rows = batch_to_rows(batch_samples, scores, threshold, per_event_scores, enc["event_meta"])
            batch_df = pd.DataFrame(batch_rows)
            keep = ~(
                (batch_df["host"].astype(str) == "unknown")
                & (batch_df["timestamp"].astype(str) == "1970-01-01T00:00:00")
                & (batch_df["anomaly_score"].astype(float) == 0.0)
            )
            removed = int((~keep).sum())
            if removed > 0:
                print(f"[EventDetector] Removed {removed} invalid placeholder rows before evaluation")
            all_rows.extend(batch_df.loc[keep].to_dict("records"))
            score_tensor = res["score"].cpu()
            for row_idx, keep_i in enumerate(keep.tolist()):
                if keep_i:
                    all_labels.append(int(batch_samples[row_idx].get("label", 0)))
                    all_scores.append(score_tensor[row_idx : row_idx + 1])
    df = pd.DataFrame(all_rows)
    if all_scores:
        score_tensor = torch.cat(all_scores, dim=0)
    else:
        score_tensor = torch.zeros(0)
    return score_tensor, np.array(all_labels), df


def summarize_threshold_sweep(labels, scores, base_threshold, save_dir):
    metrics = []
    if len(set(labels.tolist() if isinstance(labels, np.ndarray) else labels)) <= 1:
        return {}
    quantiles = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
    threshold_candidates = [float(np.quantile(scores, q)) for q in quantiles]
    threshold_candidates.append(float(base_threshold))
    threshold_candidates = sorted(set(threshold_candidates))

    from sklearn.metrics import precision_recall_fscore_support

    best = None
    for thr in threshold_candidates:
        preds = (scores > thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        row = {"threshold": float(thr), "precision": float(p), "recall": float(r), "f1": float(f)}
        metrics.append(row)
        if best is None or row["f1"] > best["f1"] or (row["f1"] == best["f1"] and row["recall"] > best["recall"]):
            best = row

    payload = {"base_threshold": float(base_threshold), "best_f1": best, "candidates": metrics}
    with open(os.path.join(save_dir, "event_threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if best is not None:
        print(
            f"[EventDetector] Best threshold by F1: {best['threshold']:.6f} "
            f"(P={best['precision']:.4f}, R={best['recall']:.4f}, F1={best['f1']:.4f})"
        )
    return payload


def build_client_partitions(dataset):
    host_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        if hasattr(dataset, "get_metadata"):
            meta = dataset.get_metadata(idx)
        else:
            sample = dataset[idx]
            meta = {"host": sample.get("host", "unknown")}
        host = str(meta.get("host", "unknown"))
        host_to_indices[host].append(idx)
    return host_to_indices


def train_event_client(local_dataset, encoder, detector, batch_size, local_epochs, lr, device, grad_dp_cfg):
    local_encoder = copy.deepcopy(encoder).to(device)
    local_detector = copy.deepcopy(detector).to(device)
    params = list(local_encoder.parameters()) + list(local_detector.parameters())
    init_vec = torch.nn.utils.parameters_to_vector(params).detach().clone()
    loader = DataLoader(
        local_dataset,
        batch_size=max(1, min(int(batch_size), len(local_dataset))),
        shuffle=True,
        collate_fn=lambda x: x,
    )
    optimizer = torch.optim.Adam(params, lr=lr)
    local_encoder.train()
    local_detector.train()
    total_loss = 0.0
    total_seen = 0
    for _ in range(int(local_epochs)):
        for batch_samples in loader:
            enc = local_encoder(batch_samples)
            loss = local_detector.reconstruction_loss(enc["event_embeddings"], enc["event_mask"])["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_samples)
            total_seen += len(batch_samples)

    update = torch.nn.utils.parameters_to_vector(params).detach().clone() - init_vec
    if grad_dp_cfg.enabled:
        clip_by_l2_norm_(update, float(grad_dp_cfg.base_clip_C0))
        add_gaussian_noise_(update, float(grad_dp_cfg.noise_sigma0))
    metrics = {"client_loss": total_loss / max(total_seen, 1), "client_samples": len(local_dataset)}
    return update.cpu(), metrics, len(local_dataset)


def run_federated_training(train_ds, encoder, detector, train_cfg, model_cfg, device, overrides=None):
    overrides = overrides or {}
    host_to_indices = build_client_partitions(train_ds)
    client_ids = sorted([cid for cid, idxs in host_to_indices.items() if idxs])
    server_cfg = ServerConfig(
        rounds=int(overrides.get("rounds", train_cfg.get("epochs", train_cfg.get("federated", {}).get("rounds", 1)))),
        client_fraction=float(overrides.get("client_fraction", train_cfg.get("client_fraction", 1.0))),
        min_clients=int(overrides.get("min_clients", train_cfg.get("min_clients", 1))),
        server_lr=float(overrides.get("server_lr", train_cfg.get("server_lr", 1.0))),
        secure_agg_enabled=bool(overrides.get("secure_agg_enabled", True)),
        secure_agg_protocol=str(overrides.get("secure_agg_protocol", "pairwise_masking")),
    )
    server = FederatedServer(client_ids, server_cfg, seed=int(train_cfg.get("seed", 42)))

    grad_cfg_raw = dict(model_cfg.get("grad_dp", {}))
    grad_dp_cfg = GradDPConfig(
        enabled=bool(overrides.get("enable_privacy", grad_cfg_raw.get("enabled", True))),
        base_clip_C0=float(overrides.get("dp_clip", grad_cfg_raw.get("base_clip_C0", 1.0))),
        importance_alpha=float(grad_cfg_raw.get("importance_alpha", 0.5)),
        noise_sigma0=float(overrides.get("dp_noise", grad_cfg_raw.get("noise_sigma0", 0.05))),
        noise_schedule=grad_cfg_raw.get("noise_schedule"),
    )
    client_cfg = train_cfg.get("client", {})
    local_epochs = int(overrides.get("local_epochs", client_cfg.get("local_epochs", 1)))
    batch_size = int(overrides.get("batch_size", client_cfg.get("batch_size", train_cfg.get("batch_size", 8))))
    lr = float(overrides.get("lr", client_cfg.get("lr", 1e-4)))

    params = list(encoder.parameters()) + list(detector.parameters())
    history = []
    for round_idx in range(server_cfg.rounds):
        selected = server.sample_clients()
        updates, ns, metrics = [], [], []
        for client_id in tqdm(selected, desc=f"FL Round {round_idx + 1}/{server_cfg.rounds}"):
            local_subset = Subset(train_ds, host_to_indices[client_id])
            update, metric, n = train_event_client(local_subset, encoder, detector, batch_size, local_epochs, lr, device, grad_dp_cfg)
            updates.append(update)
            ns.append(n)
            metrics.append(metric)
        server.aggregate_and_apply(params, updates, ns, metrics=metrics)
        avg_loss = float(np.mean([m["client_loss"] for m in metrics])) if metrics else 0.0
        print(
            f"[EventDetector][FL] Round {round_idx + 1}/{server_cfg.rounds} "
            f"clients={len(selected)} avg_loss={avg_loss:.6f}"
        )
        history.append({"round": round_idx + 1, "clients": len(selected), "avg_client_loss": avg_loss})
    return history


def train_event_detector(
    config_path,
    event_config_path=None,
    epochs_override=None,
    batch_size_override=None,
    max_seq_len_override=None,
    federated_override=None,
    rounds_override=None,
    client_fraction_override=None,
    min_clients_override=None,
):
    optc_cfg, model_cfg, train_cfg, event_cfg, device, resolved_event_path = setup_environment(
        config_path,
        event_config_path=event_config_path,
    )
    cache_dir = optc_cfg["cache_dir"]
    detector_prefix = optc_cfg.get("detector_prefix", "detector")
    test_prefix = optc_cfg.get("test_prefix", "test")
    event_model_cfg = event_cfg.get("model", {})
    event_train_cfg = event_cfg.get("training", {})
    event_fed_cfg = event_cfg.get("federated", {})
    print(f"[EventDetector] Event config: {resolved_event_path}")

    aux_map = load_cached_aux(cache_dir)
    train_cache_reader = EventCacheReader(cache_dir, detector_prefix)
    test_cache_reader = EventCacheReader(cache_dir, test_prefix)
    if train_cache_reader.cache_is_dirty or test_cache_reader.cache_is_dirty:
        raise RuntimeError(
            "Event cache is stale or incomplete. Rebuild it with "
            "`python scripts/make_event_cache.py --config configs/final_production.yaml --splits detector,test --samples_per_shard 32` "
            "before training."
        )
    if len(train_cache_reader) > 0 and len(test_cache_reader) > 0:
        train_ds = EventCachedDataset(len(train_cache_reader), aux_map, train_cache_reader)
        test_ds = EventCachedDataset(len(test_cache_reader), aux_map, test_cache_reader)
        print(f"[EventDetector] Using sharded event cache: train={len(train_cache_reader)} test={len(test_cache_reader)}")
    else:
        train_ds = EventAugmentedDataset(OpTCEcarDataset(cache_dir, split=detector_prefix, preload=False), aux_map)
        test_ds = EventAugmentedDataset(OpTCEcarDataset(cache_dir, split=test_prefix, preload=False), aux_map)

    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else event_train_cfg.get("batch_size", train_cfg.get("batch_size", 8))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    train_score_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    max_seq_len = int(
        max_seq_len_override
        if max_seq_len_override is not None
        else event_model_cfg.get("max_seq_len", model_cfg.get("max_seq_len", 128))
    )

    encoder = EventWindowEncoder(
        event_dim=int(event_model_cfg.get("event_dim", model_cfg.get("target_dim", 128))),
        num_subspaces=int(event_model_cfg.get("num_subspaces", model_cfg.get("num_subspaces", 4))),
        max_events=max_seq_len,
    ).to(device)
    detector = EventMaskedReconstructionDetector(
        event_dim=int(event_model_cfg.get("event_dim", model_cfg.get("target_dim", 128))),
        num_layers=int(event_model_cfg.get("detector_layers", 2)),
        num_heads=int(event_model_cfg.get("detector_heads", 4)),
        mask_ratio=float(event_model_cfg.get("mask_ratio", model_cfg.get("event_mask_ratio", 0.2))),
        topk_fraction=float(event_model_cfg.get("topk_fraction", model_cfg.get("event_topk_fraction", 0.1))),
        chunk_size=max_seq_len,
        inference_mask_batch=int(event_model_cfg.get("inference_mask_batch", model_cfg.get("event_inference_mask_batch", 16))),
    ).to(device)

    epochs = int(
        epochs_override
        if epochs_override is not None
        else event_train_cfg.get("epochs", train_cfg.get("detector_epochs", 20))
    )
    use_federated = bool(
        federated_override
        if federated_override is not None
        else train_cfg.get("federated_learning", False)
    )

    if use_federated:
        history = run_federated_training(
            train_ds,
            encoder,
            detector,
            train_cfg,
            deep_update(model_cfg, {"grad_dp": event_fed_cfg.get("grad_dp", {})}),
            device,
            overrides={
                "rounds": rounds_override if rounds_override is not None else event_fed_cfg.get("rounds"),
                "client_fraction": client_fraction_override if client_fraction_override is not None else event_fed_cfg.get("client_fraction"),
                "min_clients": min_clients_override if min_clients_override is not None else event_fed_cfg.get("min_clients"),
                "batch_size": batch_size,
                "local_epochs": event_fed_cfg.get("local_epochs"),
                "lr": event_fed_cfg.get("lr"),
                "server_lr": event_fed_cfg.get("server_lr"),
                "enable_privacy": event_fed_cfg.get("enable_privacy"),
                "dp_clip": event_fed_cfg.get("dp_clip"),
                "dp_noise": event_fed_cfg.get("dp_noise"),
                "secure_agg_enabled": event_fed_cfg.get("secure_agg_enabled"),
                "secure_agg_protocol": event_fed_cfg.get("secure_agg_protocol"),
            },
        )
    else:
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(detector.parameters()),
            lr=float(event_train_cfg.get("lr", 1e-4)),
        )
        encoder.train()
        detector.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            total_seen = 0
            for batch_samples in tqdm(train_loader, desc=f"Event Detector Epoch {epoch + 1}/{epochs}"):
                enc = encoder(batch_samples)
                loss_dict = detector.reconstruction_loss(enc["event_embeddings"], enc["event_mask"])
                loss = loss_dict["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_samples)
                total_seen += len(batch_samples)
            epoch_loss = total_loss / max(total_seen, 1)
            history.append({"epoch": epoch + 1, "loss": epoch_loss})
            print(f"[EventDetector] Epoch {epoch + 1}/{epochs} loss={epoch_loss:.6f}")

    benign_scores, _, _ = compute_scores(train_score_loader, encoder, detector, desc="Scoring Train")
    quantile = float(event_train_cfg.get("threshold_quantile", train_cfg.get("threshold_quantile", 0.99)))
    detector.fit_stats(benign_scores.to(device), quantile=quantile)
    print(
        f"[EventDetector] Stats fitted: mean={detector.mean_score.item():.6f}, "
        f"std={detector.std_score.item():.6f}, threshold={detector.threshold.item():.6f}"
    )

    scores, labels, df = compute_scores(test_loader, encoder, detector, desc="Scoring Test")
    save_dir = os.path.join("experiments", "event_ad_v1_fed" if use_federated else "event_ad_v1")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "detector": detector.state_dict(),
            "federated": use_federated,
        },
        os.path.join(save_dir, "event_detector.pt"),
    )
    with open(os.path.join(save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    out_path = os.path.join(save_dir, "event_detection_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[EventDetector] Results saved to {out_path}")

    try:
        import evaluate

        metrics = evaluate.run_evaluation(out_path, csv_in_edt=False)
        with open(os.path.join(save_dir, "event_detection_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[EventDetector] evaluate.py auto-run skipped: {exc}")

    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    threshold = float(detector.threshold.item())
    preds = (scores.cpu().numpy() > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    auc = roc_auc_score(labels, scores.cpu().numpy()) if len(set(labels)) > 1 else 0.5
    ap = average_precision_score(labels, scores.cpu().numpy()) if len(set(labels)) > 1 else 0.0
    print(f"[EventDetector] Precision={p:.4f} Recall={r:.4f} F1={f:.4f} AUC={auc:.4f} AUPRC={ap:.4f}")
    summarize_threshold_sweep(labels, scores.cpu().numpy(), threshold, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_production.yaml")
    parser.add_argument("--event_config", default="configs/event_method.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--federated", action="store_true")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--client_fraction", type=float, default=None)
    parser.add_argument("--min_clients", type=int, default=None)
    args = parser.parse_args()
    train_event_detector(
        args.config,
        event_config_path=args.event_config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        max_seq_len_override=args.max_seq_len,
        federated_override=True if args.federated else None,
        rounds_override=args.rounds,
        client_fraction_override=args.client_fraction,
        min_clients_override=args.min_clients,
    )

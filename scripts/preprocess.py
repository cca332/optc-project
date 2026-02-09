import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import multiprocessing
import gzip
import shutil

# 临时修改 sys.path 以确保能导入 src 模块
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from optc_uras.data.raw_reader import OptcRawReader

# 设置数据路径
DATA_DIR = Path("/root/optc-project/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def save_samples_pt(samples: List[Dict[str, Any]], out_path: Path):
    """保存样本列表到 .pt 文件"""
    if not samples:
        return
    print(f"  Saving {len(samples)} samples to {out_path}...")
    torch.save(samples, out_path)
    print(f"  Saved size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

def shard_file(file_path: Path, lines_per_shard: int = 100000) -> List[Path]:
    """
    将大文件切分为小文件，以便并行处理。
    如果 shards 目录已存在且不为空，则直接返回现有分片。
    """
    # 如果文件小于 100MB，不切分
    if file_path.stat().st_size < 100 * 1024 * 1024:
        return [file_path]

    shard_dir = file_path.parent / f"{file_path.name}.shards"
    shard_dir.mkdir(exist_ok=True)
    
    # 检查是否已经切分过
    existing_shards = list(shard_dir.glob("part_*.json.gz"))
    if existing_shards:
        print(f"  Found {len(existing_shards)} existing shards for {file_path.name}, using them.")
        return sorted(existing_shards)

    print(f"  Sharding large file {file_path.name} (this may take a while but speeds up processing)...")
    
    shards = []
    part_idx = 0
    current_lines = []
    
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f_in:
            for line in tqdm(f_in, desc="  Splitting", unit="lines"):
                current_lines.append(line)
                if len(current_lines) >= lines_per_shard:
                    # 写入分片
                    part_path = shard_dir / f"part_{part_idx:04d}.json.gz"
                    with gzip.open(part_path, "wt", encoding="utf-8") as f_out:
                        f_out.writelines(current_lines)
                    shards.append(part_path)
                    current_lines = []
                    part_idx += 1
            
            # 写入剩余部分
            if current_lines:
                part_path = shard_dir / f"part_{part_idx:04d}.json.gz"
                with gzip.open(part_path, "wt", encoding="utf-8") as f_out:
                    f_out.writelines(current_lines)
                shards.append(part_path)
                
    except Exception as e:
        print(f"  Error during sharding: {e}")
        # 如果切分失败，回退到使用原文件
        return [file_path]
        
    print(f"  Created {len(shards)} shards.")
    return shards

def process_single_file(args):
    """单个文件处理函数 (用于多进程)"""
    file_path, window_minutes = args
    try:
        # 在进程内实例化 Reader
        reader = OptcRawReader(
            file_path=str(file_path), 
            window_minutes=window_minutes
        )
        # 消耗迭代器生成列表
        return list(reader.iterate_samples())
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Preprocess OpTC raw datasets.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file (ignored in this version)")
    
    # OOM 保护：将默认并发数从 CPU 核数降级为更保守的值 (例如 8)
    # 24 核全开会导致内存爆炸 (即使是 torch.save) 且争抢 IO
    safe_workers = 8
    parser.add_argument("--workers", type=int, default=safe_workers, 
                        help=f"Number of parallel workers (default: {safe_workers})")

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Using {args.workers} worker processes for parallel processing.")

    # 定义任务配置
    tasks = [
        ("Teacher Pretrain", ["17-18day"], "teacher_pretrain.pt", False),
        ("Student Full", ["18-19day", "19day"], "student_full_temp.pt", True),
        # ("Test Set", ["25day"], "test.pt", False)
    ]

    # --- Phase 1: 并行切分所有大文件 ---
    print("\n=== Phase 1: Parallel Sharding ===")
    all_keywords = set()
    for _, kws, _, _ in tasks:
        all_keywords.update(kws)
    
    # 找到所有涉及的唯一原始文件
    unique_raw_files = set()
    for kw in all_keywords:
        found = list(RAW_DIR.glob(f"**/*{kw}*.json.gz"))
        unique_raw_files.update(found)
    
    unique_raw_files = sorted(list(unique_raw_files))
    print(f"Found {len(unique_raw_files)} unique large files to shard: {[f.name for f in unique_raw_files]}")

    # 并行执行切分
    # 切分是 IO + CPU 密集型，且每个文件很大，并发数不宜过多，以免磁盘 IO 争抢太厉害
    # 既然只有 3-4 个大文件，我们直接用 len(unique_raw_files) 个进程
    shard_workers = min(len(unique_raw_files), 8)
    if shard_workers > 0:
        print(f"Starting sharding with {shard_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=shard_workers) as executor:
            # 提交所有切分任务
            future_to_file = {executor.submit(shard_file, f): f for f in unique_raw_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    shards = future.result()
                    print(f"  Successfully sharded {f.name} into {len(shards)} parts.")
                except Exception as exc:
                    print(f"  Generated exception during sharding {f.name}: {exc}")

    print("All sharding completed.\n")

    # --- Phase 2: 处理任务 ---
    print("=== Phase 2: Processing Samples ===")
    for task_name, keywords, out_name, need_split in tasks:
        print(f"\nProcessing {task_name}...")
        
        # 1. 找到所有匹配的文件 (此时应该是已经切分好的)
        raw_files = []
        for kw in keywords:
            raw_files.extend(list(RAW_DIR.glob(f"**/*{kw}*.json.gz")))
            
        if not raw_files:
            print(f"  Warning: No files found for keywords {keywords}, skipping.")
            continue
            
        # 2. 收集所有分片
        target_files = []
        for f in raw_files:
            # 这里的 shard_file 会直接返回已存在的 shards，速度极快
            target_files.extend(shard_file(f))

        print(f"  Total {len(target_files)} file shards to process. Starting parallel processing...")

        # 3. 并行读取并处理样本 (Streaming Mode)
        total_generated = 0
        process_args = [(f, 15) for f in target_files]
        
        # 确保输出目录存在
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_file, arg) for arg in process_args]
            
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc="  Processing Shards"):
                file_samples = future.result()
                if not file_samples:
                    continue
                
                # 立即保存分片结果，不堆积在内存中
                if need_split:
                    # 针对 Student Set，在分片内部进行 85/15 划分
                    # 注意：这是对随机性的近似，但在大数据量下分布是均匀的
                    import random
                    # 为每个样本随机分配去向
                    train_chunk = []
                    val_chunk = []
                    for s in file_samples:
                        if random.random() < 0.85:
                            train_chunk.append(s)
                        else:
                            val_chunk.append(s)
                    
                    if train_chunk:
                        save_samples_pt(train_chunk, PROCESSED_DIR / f"student_train_part_{i:04d}.pt")
                    if val_chunk:
                        save_samples_pt(val_chunk, PROCESSED_DIR / f"student_val_part_{i:04d}.pt")
                    
                    total_generated += len(file_samples)
                else:
                    # 针对 Teacher Set，直接保存
                    # 使用 out_name 去掉 .pt 后缀作为前缀
                    stem = out_name.replace(".pt", "")
                    save_samples_pt(file_samples, PROCESSED_DIR / f"{stem}_part_{i:04d}.pt")
                    total_generated += len(file_samples)
                
                # 显式释放内存
                del file_samples

        print(f"  Total samples generated: {total_generated}")
        # 移除旧的整体保存逻辑
        print("\nAll Done!")

if __name__ == "__main__":
    main()

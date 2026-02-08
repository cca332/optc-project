import pandas as pd
import sys
import os

def analyze():
    # 指向 results4 目录
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results4", "detection_results.csv")
    if not os.path.exists(results_path):
        # Fallback to current dir if running from root
        results_path = "results4/detection_results.csv"
        
    if not os.path.exists(results_path):
        print(f"Waiting for pipeline to finish... File not found: {results_path}")
        return

    print(f"Loading results from {results_path}...")
    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Total test samples: {len(df)}")
    
    # 统计异常
    anomalies = df[df["is_anomaly"] == 1]
    print(f"Total anomalies detected (Threshold Based): {len(anomalies)} (Rate: {len(anomalies)/len(df):.2%})")
    
    # [DEBUG] Always show top scores to debug threshold issues
    print("\n=== Top 20 Samples by Raw Anomaly Score (Debugging) ===")
    top_20 = df.sort_values("anomaly_score", ascending=False).head(20)
    cols = ["timestamp", "host", "anomaly_score", "top_attribution"]
    available_cols = [c for c in cols if c in df.columns]
    print(top_20[available_cols].to_string(index=False))
    
    # 专门检查目标靶机 SysClient0051
    target_host = "SysClient0051"
    if "host" in df.columns:
        # Check all samples from target host, not just anomalies
        target_anoms = df[df["host"].str.contains(target_host, case=False, na=False)].copy()
        
        print(f"\n=== Target Host Analysis ({target_host}) ===")
        print(f"Total samples for target: {len(target_anoms)}")
        
        if not target_anoms.empty:
            # Sort by time
            target_anoms = target_anoms.sort_values("timestamp")
            
            # Timestamp is now already EDT (from run_pipeline.py)
            target_anoms["dt_edt"] = pd.to_datetime(target_anoms["timestamp"])
            target_anoms["est_edt"] = target_anoms["dt_edt"].dt.strftime('%H:%M:%S')

            print("\nAll Scores on Target Host (Time Sorted) [EDT]:")
            # Reorder columns to show EDT next to timestamp
            cols_to_show = ["timestamp", "est_edt", "host", "anomaly_score", "top_attribution"]
            print(target_anoms[cols_to_show].to_string(index=False))
            
            # Check Rank
            max_score = target_anoms["anomaly_score"].max()
            rank = (df["anomaly_score"] > max_score).sum() + 1
            print(f"\nHighest Rank for Target Host: #{rank} / {len(df)}")
            
            # Highlight likely attack windows
            print("\n[Analysis] Note: OpTC attacks often occur in EDT (UTC-4).")
            print("  - 13:42 EDT  => ~17:42 UTC (Window 17:30 or 17:45)")
            print("  - 14:24 EDT  => ~18:24 UTC (Window 18:15 or 18:30)")
        else:
            print("⚠️ Warning: No samples found for target host.")
    else:
        print("Warning: 'host' column not found, cannot filter by target.")

if __name__ == "__main__":
    analyze()
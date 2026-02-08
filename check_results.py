
# PEP 263 Compliance
import pandas as pd
import numpy as np

p = r'c:\Users\32546\Desktop\optc\results\detection_results.csv'
try:
    df = pd.read_csv(p)
    df['ts'] = pd.to_datetime(df['timestamp'])

    # Stats
    total = df[df.is_anomaly==1].shape[0]
    target_mask = df.host.str.contains('0051', na=False) & (df.is_anomaly==1)
    target = df[target_mask].shape[0]
    
    # 18:00 False Positives
    fp_mask = (df.ts.dt.hour==18) & (df.is_anomaly==1) & (~df.host.str.contains('0051', na=False))
    fps = df[fp_mask].shape[0]

    print('-'*30)
    print(f'Total Anomalies: {total}')
    print(f'SysClient0051 (Target) Caught: {target}')
    print(f'18:00 False Positives: {fps}')
    print('-'*30)
    
    # Check for anomaly_prob column
    print(f"Columns: {list(df.columns)}")
    if 'anomaly_prob' in df.columns:
        print("FAIL: anomaly_prob column still exists!")
    else:
        print("PASS: anomaly_prob column removed.")

    # Debug Stats
    if target > 0:
        print(f"Target Mean Score: {df[target_mask].anomaly_score.mean():.4e}")
        print(f"Target Mean Thresh: {df[target_mask].adaptive_threshold.mean():.4e}")
    else:
        # Check target scores anyway (even if not caught)
        t_all_mask = df.host.str.contains('0051', na=False)
        if t_all_mask.sum() > 0:
            print(f"Target (All) Mean Score: {df[t_all_mask].anomaly_score.mean():.4e}")
            print(f"Target (All) Mean Thresh: {df[t_all_mask].adaptive_threshold.mean():.4e}")

    if fps > 0:
        print(f"FP Mean Score: {df[fp_mask].anomaly_score.mean():.4e}")
        print(f"FP Mean Thresh: {df[fp_mask].adaptive_threshold.mean():.4e}")
    
    print('-'*30)

except Exception as e:
    print(f"Error: {e}")


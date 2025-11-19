import pandas as pd
import numpy as np
import json
import os
import sys
from scipy import stats

# Configuration
LOG_FILE = 'prediction_log.csv'
BASELINE_CONFIDENCE_THRESHOLD = 0.70  # If avg confidence drops below this, alert!
MIN_SAMPLES_FOR_DRIFT = 10  # Need at least 10 logs to be statistically significant

def check_drift():
    print("="*60)
    print("🔍 MODEL DRIFT MONITORING SYSTEM")
    print("="*60)

    # 1. Load Logs
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Log file '{LOG_FILE}' not found.")
        print("   -> Go to the website and run some predictions first!")
        return

    try:
        df_log = pd.read_csv(LOG_FILE)
        print(f"✓ Loaded {len(df_log)} production logs.")
    except Exception as e:
        print(f"❌ Error reading logs: {e}")
        return

    # 2. Volume Check
    if len(df_log) < MIN_SAMPLES_FOR_DRIFT:
        print(f"⚠️  Not enough data for drift detection.")
        print(f"   Current: {len(df_log)} | Required: {MIN_SAMPLES_FOR_DRIFT}")
        print("   -> Please generate more predictions on the website.")
        return

    # 3. Parse Confidence (Handle potential string formatting)
    # Confidence might be stored as strings "0.95" or floats 0.95
    try:
        df_log['confidence'] = pd.to_numeric(df_log['confidence'], errors='coerce')
        current_avg_conf = df_log['confidence'].mean()
    except Exception as e:
        print(f"⚠️ Could not calculate confidence stats: {e}")
        current_avg_conf = 0

    # 4. Check for "Concept Drift" (Model Uncertainty)
    print("\n📊 STATISTICAL ANALYSIS")
    print(f"   Average Model Confidence: {current_avg_conf:.4f}")
    
    if current_avg_conf < BASELINE_CONFIDENCE_THRESHOLD:
        print(f"   🔴 ALERT: Model Confidence is LOW (< {BASELINE_CONFIDENCE_THRESHOLD})")
        print("      The model is struggling to classify recent data.")
        print("      Reason: New phishing patterns may have emerged.")
    else:
        print(f"   ✅ Status: Healthy (High Confidence)")

    # 5. Check for "Label Drift" (Output Distribution)
    # Are we predicting way more Phishing than usual?
    phishing_count = len(df_log[df_log['prediction'] == 'Phishing'])
    safe_count = len(df_log[df_log['prediction'] == 'Safe'])
    total = phishing_count + safe_count
    
    phishing_ratio = phishing_count / total
    
    print(f"\n📈 PREDICTION DISTRIBUTION")
    print(f"   Phishing: {phishing_count} ({phishing_ratio:.1%})")
    print(f"   Safe:     {safe_count} ({1-phishing_ratio:.1%})")

    # A simple heuristic: If 100% of traffic is one class, something might be wrong
    if phishing_ratio > 0.9 or phishing_ratio < 0.1:
        print("   ⚠️  WARNING: Extreme Class Imbalance detected.")
        print("      If this is real traffic, it's suspicious. Check the inputs.")

    # 6. Input Type Breakdown
    if 'input_type' in df_log.columns:
        print("\n📧 INPUT SOURCES")
        print(df_log['input_type'].value_counts().to_string())

    print("\n" + "="*60)
    print("✅ DRIFT CHECK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    check_drift()
import pandas as pd
import numpy as np
import os
import requests
import json

# ==========================================
# CONFIGURATION
# ==========================================
LOG_FILE = 'prediction_log.csv'
BASELINE_CONFIDENCE_THRESHOLD = 0.70
MIN_SAMPLES_FOR_DRIFT = 10

# GitHub Configuration (Reads from Docker Environment)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
REPO_OWNER = "JasmithaTholeti"
REPO_NAME = "Phishing_detection_CI-CD"
WORKFLOW_ID = "main_pipeline.yml"  # Updated to match your new filename

def trigger_retraining():
    """Triggers the GitHub Action Workflow via API"""
    print("\n🚀 TRIGGERING AUTOMATED RETRAINING...")
    
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN not found. Cannot trigger pipeline.")
        return

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_ID}/dispatches"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "ref": "main"
    }
    
    try:
        print(f"   Target URL: {url}")
        response = requests.post(url, headers=headers, json=data)
        
        # --- DEBUG INFO ---
        print(f"   GitHub API Response Code: {response.status_code}")
        if response.status_code != 204:
            print(f"   GitHub API Error Body: {response.text}")
        # ------------------

        if response.status_code == 204:
            print("✅ Successfully triggered GitHub Pipeline!")
            print("   Check your Actions tab in a few seconds.")
        else:
            print(f"❌ Failed to trigger pipeline.")
            
    except Exception as e:
        print(f"❌ Error calling GitHub API: {e}")

def check_drift():
    print("="*60)
    print("🔍 MODEL DRIFT MONITORING SYSTEM")
    print("="*60)

    # 1. Load Logs
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Log file '{LOG_FILE}' not found.")
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
        return

    # 3. Calculate Average Confidence
    try:
        df_log['confidence'] = pd.to_numeric(df_log['confidence'], errors='coerce')
        current_avg_conf = df_log['confidence'].mean()
    except Exception as e:
        print(f"⚠️ Could not calculate confidence stats: {e}")
        current_avg_conf = 0

    # 4. Drift Logic
    print("\n📊 STATISTICAL ANALYSIS")
    print(f"   Average Model Confidence: {current_avg_conf:.4f}")
    
    if current_avg_conf < BASELINE_CONFIDENCE_THRESHOLD:
        print(f"   🔴 ALERT: Drift Detected! Confidence is below {BASELINE_CONFIDENCE_THRESHOLD}")
        trigger_retraining()
    else:
        print(f"   ✅ Status: Healthy (High Confidence)")

    # 5. Prediction Distribution
    phishing_count = len(df_log[df_log['prediction'] == 'Phishing'])
    total = len(df_log)
    phishing_ratio = phishing_count / total if total > 0 else 0
    
    print(f"\n📈 PREDICTION DISTRIBUTION")
    print(f"   Phishing: {phishing_count} ({phishing_ratio:.1%})")
    print(f"   Safe:     {total - phishing_count} ({1-phishing_ratio:.1%})")

    if 'input_type' in df_log.columns:
        print("\n📧 INPUT SOURCES")
        print(df_log['input_type'].value_counts().to_string())

    print("\n" + "="*60)
    print("✅ DRIFT CHECK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    check_drift()
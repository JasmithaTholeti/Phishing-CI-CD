import pandas as pd
import numpy as np
import os
import requests  # <--- You might need to pip install requests
import json

# Configuration
LOG_FILE = 'prediction_log.csv'
BASELINE_CONFIDENCE_THRESHOLD = 0.70
MIN_SAMPLES_FOR_DRIFT = 10

# GitHub Configuration (Replace these!)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # In production, use os.getenv('GITHUB_TOKEN')
REPO_OWNER = "JasmithaTholeti"
REPO_NAME = "Phishing_detection_CI-CD"
WORKFLOW_ID = "ci.yml"  # The filename of your workflow

def trigger_retraining():
    """Triggers the GitHub Action Workflow via API"""
    print("\n🚀 TRIGGERING AUTOMATED RETRAINING...")
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_ID}/dispatches"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "ref": "main"  # The branch you want to run
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 204:
            print("✅ Successfully triggered GitHub Pipeline!")
            print("   Check your Actions tab in a few seconds.")
        else:
            print(f"❌ Failed to trigger pipeline: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error calling GitHub API: {e}")

def check_drift():
    print("="*60)
    print("🔍 MODEL DRIFT MONITORING SYSTEM")
    print("="*60)

    # ... (Loading Logic remains the same) ...
    if not os.path.exists(LOG_FILE):
        return

    try:
        df_log = pd.read_csv(LOG_FILE)
    except:
        return

    if len(df_log) < MIN_SAMPLES_FOR_DRIFT:
        print(f"⚠️  Not enough data ({len(df_log)} samples).")
        return

    # Calculate Confidence
    try:
        df_log['confidence'] = pd.to_numeric(df_log['confidence'], errors='coerce')
        current_avg_conf = df_log['confidence'].mean()
    except:
        current_avg_conf = 0

    print(f"\n📊 Current Confidence: {current_avg_conf:.4f}")
    
    # --- THE TRIGGER LOGIC ---
    if current_avg_conf < BASELINE_CONFIDENCE_THRESHOLD:
        print(f"   🔴 ALERT: Drift Detected! Confidence is below {BASELINE_CONFIDENCE_THRESHOLD}")
        
        # AUTOMATICALLY TRIGGER RETRAINING
        trigger_retraining()
        
    else:
        print(f"   ✅ Status: Healthy")

if __name__ == "__main__":
    check_drift()
```

### Step 4: Verify your YAML
Check your `.github/workflows/ci.yml`. Look at the top.
Does it have `workflow_dispatch:`?

```yaml
on:
  push:
    branches: [ main ]
  workflow_dispatch: {}  # <--- THIS MUST BE HERE
```
*If this line exists, your pipeline allows manual/API triggers.*

---

### How to Test It
1.  **Generate "Bad" Traffic:** Paste weird text into your app to lower the confidence score.
2.  **Run the Script:**
    ```powershell
    python check_drift.py
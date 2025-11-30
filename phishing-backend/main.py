from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import atexit
from csv import DictWriter
from typing import Optional, Dict, Any

# 1. Initialize App
app = FastAPI(title="Hybrid Phishing Detection API", version="2.0")

# CORS
origins = ["http://localhost:3000","https://phishing-detection-ci-cd.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Artifacts
try:
    model = joblib.load("phishing_model.joblib")
    model_columns = joblib.load("model_columns.joblib")
    email_scaler = joblib.load("email_scaler.joblib")
    print("✓ Model, columns, and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"✗ Error loading artifacts: {e}")
    print("  Please run 'train-hybrid.py' first to generate .joblib files.")
    model = None
    model_columns = []
    email_scaler = None

# --- LOGGING SETUP (Restored for Drift Detection) ---
LOG_FILE = 'prediction_log.csv'

# Determine Log Columns
if model_columns:
    LOG_COLUMNS = list(model_columns) + ['prediction', 'confidence', 'input_type']
else:
    LOG_COLUMNS = ['prediction', 'confidence', 'input_type'] # Fallback

# Initialize Log File
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()

log_file_writer = open(LOG_FILE, 'a', newline='')
log_writer = DictWriter(log_file_writer, fieldnames=LOG_COLUMNS)

@atexit.register
def close_log_file():
    print("Closing log file...")
    log_file_writer.close()
# --- END LOGGING SETUP ---


# 3. Define Logic - MUST match train-hybrid.py exactly
def extract_email_features(text):
    """Extract numerical features from email text"""
    if pd.isna(text) or text == '':
        text = ''
    text = str(text).lower()
    features = {}
    
    # 1. Length-based
    features['email_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # 2. Keywords
    phishing_keywords = ['urgent', 'verify', 'account', 'suspended', 'click', 'confirm', 
                        'password', 'credit', 'bank', 'security', 'update', 'expire', 
                        'winner', 'prize', 'congratulations', 'claim', 'free', 'offer']
    features['suspicious_keywords'] = sum(keyword in text for keyword in phishing_keywords)
    
    # 3. URL & Char counts
    features['has_url'] = 1 if re.search(r'http[s]?://', text) else 0
    features['url_count'] = len(re.findall(r'http[s]?://', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['dollar_sign'] = text.count('$')
    features['at_symbol'] = text.count('@')
    
    # 4. Complexity
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['has_numbers'] = 1 if re.search(r'\d', text) else 0
    features['number_count'] = len(re.findall(r'\d+', text))
    features['email_addresses'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    
    # 5. Context
    urgent_words = ['urgent', 'immediate', 'action required', 'act now', 'limited time']
    features['urgency_score'] = sum(word in text for word in urgent_words)
    
    financial_terms = ['money', 'payment', 'transfer', 'account', 'bank', 'credit card']
    features['financial_terms'] = sum(term in text for term in financial_terms)
    
    generic_greetings = ['dear customer', 'dear user', 'dear member', 'valued customer']
    features['generic_greeting'] = 1 if any(greeting in text for greeting in generic_greetings) else 0
    
    return features

# 4. Pydantic Models
class HybridInput(BaseModel):
    email_text: Optional[str] = None
    website_features: Optional[Dict[str, int]] = None

@app.post("/predict")
def predict_hybrid(data: HybridInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = None
    input_type = "Unknown"
    
    # --- PATH A: Process Email Text ---
    if data.email_text is not None:
        input_type = "Email"
        # 1. Extract features
        raw_features = extract_email_features(data.email_text)
        features_df = pd.DataFrame([raw_features])
        
        # 2. Normalize using the loaded scaler
        try:
            if hasattr(email_scaler, 'feature_names_in_'):
                features_df = features_df[email_scaler.feature_names_in_]
            
            scaled_array = email_scaler.transform(features_df)
            features_df = pd.DataFrame(scaled_array, columns=features_df.columns)
        except Exception as e:
            return {"error": f"Scaling failed. Feature mismatch? {str(e)}"}

        # 3. Binning
        for col in features_df.columns:
            features_df[col] = pd.cut(
                features_df[col], 
                bins=[-np.inf, -0.5, 0.5, np.inf], 
                labels=[-1, 0, 1]
            ).astype(int)

        features_df['source_website'] = 0
        input_df = features_df

    # --- PATH B: Process Website Features ---
    elif data.website_features is not None:
        input_type = "Website"
        features_df = pd.DataFrame([data.website_features])
        features_df.columns = features_df.columns.str.lower()
        features_df['source_website'] = 1
        input_df = features_df

    else:
        raise HTTPException(status_code=400, detail="Please provide either 'email_text' or 'website_features'")

    # --- MERGE & PREDICT ---
    try:
        # 1. Align columns (Fill missing with 0)
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # 2. Reorder columns
        input_df = input_df[model_columns]

        # 3. Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        result = "Phishing" if prediction[0] == 1 else "Safe"
        confidence_val = float(probability[0][prediction[0]])

        # --- LOGGING (Module 7) ---
        # We log the FULL input vector (features) + prediction
        try:
            log_entry = input_df.iloc[0].to_dict()
            log_entry['prediction'] = result
            log_entry['confidence'] = f"{confidence_val:.4f}"
            log_entry['input_type'] = input_type
            
            log_writer.writerow(log_entry)
            log_file_writer.flush()
        except Exception as e:
            print(f"Logging failed: {e}")
        # --- END LOGGING ---

        return {
            "prediction": result,
            "confidence": f"{confidence_val:.4f}",
            "type": f"{input_type} Analysis"
        }

    except Exception as e:
        return {"error": f"Prediction processing failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import uvicorn

# # --------------------------------------------------
# # 🔥 FASTAPI APP
# # --------------------------------------------------
# app = FastAPI(title="Email Phishing Detection API", version="1.0")

# # --------------------------------------------------
# # 🔥 ENABLE CORS (IMPORTANT to fix 405 error)
# # --------------------------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],              # allow all for now (you can restrict later)
#     allow_credentials=True,
#     allow_methods=["*"],              # ⭐ allows POST + OPTIONS (fixes your issue)
#     allow_headers=["*"],
# )

# # --------------------------------------------------
# # 🔥 LOAD MODEL + TF-IDF VECTORIZER
# # --------------------------------------------------
# try:
#     model = joblib.load("phishing_model.joblib")
#     vectorizer = joblib.load("tfidf_vectorizer.joblib")
#     print("✓ Model + Vectorizer Loaded Successfully")
# except Exception as e:
#     print(f"❌ Failed to load model/vectorizer: {e}")
#     model = None
#     vectorizer = None

# # --------------------------------------------------
# # 🔥 INPUT FORMAT FOR API
# # --------------------------------------------------
# class EmailRequest(BaseModel):
#     email_text: str

# # --------------------------------------------------
# # 🔥 MAIN PREDICTION ROUTE
# # --------------------------------------------------
# @app.post("/predict")
# def predict_email(data: EmailRequest):

#     if model is None or vectorizer is None:
#         return {"error": "Model or vectorizer not loaded. Train model first."}

#     # 1) Convert to list for vectorizer
#     email = [data.email_text]

#     # 2) TF-IDF vectorization (same as training)
#     email_vectors = vectorizer.transform(email)

#     # 3) Predict
#     prediction = model.predict(email_vectors)[0]
#     probabilities = model.predict_proba(email_vectors)[0]

#     result = "Phishing" if prediction == 1 else "Legitimate"
#     confidence = float(max(probabilities)) * 100

#     return {
#         "prediction": result,
#         "confidence": round(confidence, 2),
#         "phishing_probability": round(probabilities[1] * 100, 2),
#         "legitimate_probability": round(probabilities[0] * 100, 2),
#         "type": "Email Analysis"
#     }

# # --------------------------------------------------
# # 🔥 RUN SERVER
# # --------------------------------------------------
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
from typing import Optional, Dict, List

# --------------------------------------------------
# 🔥 FASTAPI APP
# --------------------------------------------------
app = FastAPI(title="Phishing Detection API - Hybrid System", version="2.0")

# --------------------------------------------------
# 🔥 ENABLE CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 🔥 LOAD MODELS
# --------------------------------------------------
email_model = None
email_vectorizer = None
website_model = None
website_features = None
important_features = None

try:
    email_model = joblib.load("phishing_model.joblib")
    email_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    print("✓ Email Model + Vectorizer Loaded Successfully")
except Exception as e:
    print(f"⚠️  Email model not found: {e}")

try:
    website_model = joblib.load("phishing_website_model.joblib")
    website_features = joblib.load("website_features.joblib")
    try:
        important_features = joblib.load("important_features.joblib")
    except:
        important_features = website_features[:15]  # Default to first 15
    print("✓ Website Model + Features Loaded Successfully")
except Exception as e:
    print(f"⚠️  Website model not found: {e}")

# --------------------------------------------------
# 🔥 PHISHING DETECTION CONSTANTS
# --------------------------------------------------
PHISHING_KEYWORDS = [
    'verify', 'account', 'update', 'secure', 'banking', 'login', 'signin',
    'confirm', 'suspend', 'restrict', 'urgent', 'immediately', 'click',
    'paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'wallet', 'payment', 'billing'
]

LEGITIMATE_TLDS = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'int']
SUSPICIOUS_TLDS = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'date', 'wang']

# --------------------------------------------------
# 🔥 HELPER FUNCTIONS FOR URL ANALYSIS
# --------------------------------------------------
def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    return -sum((count/length) * np.log2(count/length) for count in counter.values())

def extract_url_features(url: str) -> Dict:
    """
    Advanced feature extraction from URL for phishing detection
    """
    features = {}
    
    try:
        # Parse URL
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        full_url = url.lower()
        
        # Remove www. for analysis
        domain_clean = domain.replace('www.', '')
        
        # ==================== CRITICAL FEATURES ====================
        
        # 1. IP Address Detection
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        has_ip = bool(re.search(ip_pattern, domain))
        features['having_ip_address'] = 1 if has_ip else -1
        
        # 2. @ Symbol
        features['having_at_symbol'] = 1 if '@' in url else -1
        
        # 3. URL Length
        url_len = len(url)
        if url_len < 54:
            features['url_length'] = -1
        elif url_len <= 75:
            features['url_length'] = 0
        else:
            features['url_length'] = 1
        
        # 4. URL Shorteners
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 
                      'buff.ly', 'adf.ly', 'short.link', 'tiny.cc']
        is_shortener = any(s in domain for s in shorteners)
        features['shortining_service'] = 1 if is_shortener else -1
        
        # 5. Prefix/Suffix with Dash
        has_dash = '-' in domain_clean
        features['prefix_suffix'] = 1 if has_dash else -1
        
        # 6. Subdomains
        dot_count = domain.count('.')
        if dot_count <= 1:
            features['having_sub_domain'] = -1
        elif dot_count == 2:
            features['having_sub_domain'] = 0
        else:
            features['having_sub_domain'] = 1
        
        # 7. HTTPS and SSL
        has_https = url.startswith('https://')
        https_in_domain = 'https' in domain
        
        if has_https and not https_in_domain:
            features['sslfinal_state'] = -1
        elif has_https:
            features['sslfinal_state'] = 0
        else:
            features['sslfinal_state'] = 1
        
        # 8. HTTPS token in domain
        features['https_token'] = 1 if https_in_domain else -1
        
        # 9. Double Slash Redirecting
        double_slash_count = url.count('//')
        features['double_slash_redirecting'] = 1 if double_slash_count > 1 else -1
        
        # ==================== DOMAIN ANALYSIS ====================
        
        # 10. Abnormal URL - Check for phishing keywords
        domain_parts = domain_clean.replace('.', ' ').replace('-', ' ').split()
        suspicious_keyword_count = sum(1 for word in domain_parts if word in PHISHING_KEYWORDS)
        
        suspicious_patterns = [
            r'(\d{3,})',  # Multiple consecutive digits
            r'([a-z])\1{3,}',  # Character repeated 3+ times
            r'secure.*account',
            r'verify.*login',
            r'update.*now'
        ]
        pattern_matches = sum(1 for pattern in suspicious_patterns if re.search(pattern, domain_clean))
        
        features['abnormal_url'] = 1 if (suspicious_keyword_count >= 2 or pattern_matches >= 1) else -1
        
        # 11. Domain Registration Length
        if len(domain_clean) > 15 and dot_count <= 2:
            features['domain_registeration_length'] = -1
        else:
            features['domain_registeration_length'] = 1
        
        # 12. Port
        port_pattern = r':(\d+)'
        port_match = re.search(port_pattern, domain)
        if port_match:
            port = int(port_match.group(1))
            features['port'] = -1 if port in [80, 443] else 1
        else:
            features['port'] = -1
        
        # ==================== STATISTICAL FEATURES ====================
        
        # 13. Entropy of domain
        entropy = calculate_entropy(domain_clean.replace('.', ''))
        if entropy < 3.0:
            domain_entropy = -1
        elif entropy < 4.0:
            domain_entropy = 0
        else:
            domain_entropy = 1
        
        # 14. Digit ratio in domain
        if domain_clean:
            digit_count = sum(c.isdigit() for c in domain_clean)
            digit_ratio = digit_count / len(domain_clean)
            
            if digit_ratio == 0:
                digit_feature = -1
            elif digit_ratio < 0.15:
                digit_feature = 0
            else:
                digit_feature = 1
        else:
            digit_feature = 0
        
        # 15. TLD Analysis
        tld = domain_clean.split('.')[-1] if '.' in domain_clean else ''
        if tld in LEGITIMATE_TLDS:
            tld_feature = -1
        elif tld in SUSPICIOUS_TLDS:
            tld_feature = 1
        else:
            tld_feature = 0
        
        # 16. Special characters count
        special_chars = len(re.findall(r'[^a-zA-Z0-9.]', domain_clean))
        if special_chars == 0:
            special_feature = -1
        elif special_chars <= 1:
            special_feature = 0
        else:
            special_feature = 1
        
        # ==================== COMBINED RISK SCORE ====================
        
        risk_factors = {
            'ip_address': has_ip * 3,
            'at_symbol': (features['having_at_symbol'] == 1) * 3,
            'shortener': is_shortener * 2,
            'many_subdomains': (dot_count > 3) * 2,
            'no_https': (not has_https) * 2,
            'https_in_domain': https_in_domain * 3,
            'suspicious_keywords': suspicious_keyword_count * 1,
            'dash_in_domain': has_dash * 1,
            'high_entropy': (entropy > 4.0) * 2,
            'suspicious_tld': (tld in SUSPICIOUS_TLDS) * 2,
            'long_url': (url_len > 75) * 1,
            'many_digits': (digit_ratio > 0.2 if 'digit_ratio' in locals() else False) * 1
        }
        
        total_risk = sum(risk_factors.values())
        
        # Adjust features based on risk score
        if total_risk >= 8:
            features['statistical_report'] = 1
            features['page_rank'] = 1
            features['web_traffic'] = 1
            features['google_index'] = 1
            features['age_of_domain'] = 1
        elif total_risk >= 5:
            features['statistical_report'] = 0
            features['page_rank'] = 0
            features['web_traffic'] = 0
            features['google_index'] = 0
            features['age_of_domain'] = 0
        else:
            features['statistical_report'] = -1
            features['page_rank'] = -1
            features['web_traffic'] = -1
            features['google_index'] = -1
            features['age_of_domain'] = -1
        
        # Default values for features we can't determine from URL alone
        features['request_url'] = 0
        features['url_of_anchor'] = 0
        features['links_in_tags'] = 0
        features['sfh'] = 0
        features['submitting_to_email'] = -1
        features['redirect'] = 0
        features['on_mouseover'] = -1
        features['rightclick'] = -1
        features['popupwidnow'] = -1
        features['iframe'] = 0
        features['dnsrecord'] = -1
        features['links_pointing_to_page'] = 0
        
        # Add metadata for response
        features['_risk_score'] = total_risk
        features['_risk_factors'] = risk_factors
        features['_url_length'] = url_len
        features['_domain'] = domain
        features['_has_https'] = has_https
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return safe defaults
        if website_features:
            features = {feat: -1 for feat in website_features}
        features['_risk_score'] = 0
        features['_error'] = str(e)
    
    return features

# --------------------------------------------------
# 🔥 INPUT MODELS
# --------------------------------------------------
class EmailRequest(BaseModel):
    email_text: str

class WebsiteRequest(BaseModel):
    url: str

class WebsiteFeaturesRequest(BaseModel):
    website_features: Dict[str, int]

# --------------------------------------------------
# 🔥 MAIN PREDICTION ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Phishing Detection API - Hybrid System",
        "version": "2.0",
        "endpoints": {
            "email": "/predict (POST)",
            "website_url": "/predict_website (POST)",
            "website_features": "/predict (POST with website_features)",
            "health": "/health (GET)"
        },
        "models_loaded": {
            "email": email_model is not None,
            "website": website_model is not None
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "email_model": "loaded" if email_model else "not loaded",
        "website_model": "loaded" if website_model else "not loaded"
    }

@app.post("/predict")
def predict(data: dict):
    """
    Unified prediction endpoint
    Accepts either:
    - {"email_text": "..."} for email analysis
    - {"website_features": {...}} for website feature analysis
    """
    
    # Email Analysis
    if "email_text" in data:
        if email_model is None or email_vectorizer is None:
            raise HTTPException(status_code=503, detail="Email model not loaded")
        
        email = [data["email_text"]]
        email_vectors = email_vectorizer.transform(email)
        
        prediction = email_model.predict(email_vectors)[0]
        probabilities = email_model.predict_proba(email_vectors)[0]
        
        result = "Phishing" if prediction == 1 else "Legitimate"
        confidence = float(max(probabilities))
        
        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "phishing_probability": round(float(probabilities[1]), 4),
            "legitimate_probability": round(float(probabilities[0]), 4),
            "type": "Email Analysis"
        }
    
    # Website Feature Analysis
    elif "website_features" in data:
        if website_model is None or website_features is None:
            raise HTTPException(status_code=503, detail="Website model not loaded")
        
        features = data["website_features"]
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feat in website_features:
            if feat not in feature_df.columns:
                feature_df[feat] = -1
        
        feature_df = feature_df[website_features]
        
        prediction = website_model.predict(feature_df)[0]
        probabilities = website_model.predict_proba(feature_df)[0]
        
        result = "Phishing" if prediction == 1 else "Legitimate"
        confidence = float(max(probabilities))
        
        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "phishing_probability": round(float(probabilities[1]), 4),
            "legitimate_probability": round(float(probabilities[0]), 4),
            "type": "Website Feature Analysis"
        }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid request. Provide 'email_text' or 'website_features'")

@app.post("/predict_website")
@app.post("/predict_website")
def predict_website(data: WebsiteRequest):
    """
    Predict phishing from URL
    Automatically extracts features and analyzes
    """
    if website_model is None or website_features is None:
        raise HTTPException(status_code=503, detail="Website model not loaded. Train model first.")
    
    url = data.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    # Extract features from URL
    extracted_features = extract_url_features(url)
    
    # Separate metadata from features
    risk_score = extracted_features.pop('_risk_score', 0)
    risk_factors = extracted_features.pop('_risk_factors', {})
    url_length = extracted_features.pop('_url_length', 0)
    domain = extracted_features.pop('_domain', '')
    has_https = extracted_features.pop('_has_https', False)
    error = extracted_features.pop('_error', None)
    
    if error:
        raise HTTPException(status_code=400, detail=f"Error extracting features: {error}")
    
    # Create DataFrame for prediction
    feature_df = pd.DataFrame([extracted_features])
    
    # Ensure all required features are present
    for feat in website_features:
        if feat not in feature_df.columns:
            feature_df[feat] = -1
    
    feature_df = feature_df[website_features]
    
    # Make prediction
    prediction = website_model.predict(feature_df)[0]
    probabilities = website_model.predict_proba(feature_df)[0]
    
    result = "Phishing" if prediction == 1 else "Legitimate"
    confidence = float(max(probabilities))
    
    # Identify warning signs
    warning_signs = []
    if extracted_features.get('having_ip_address', -1) == 1:
        warning_signs.append("IP address in URL")
    if extracted_features.get('having_at_symbol', -1) == 1:
        warning_signs.append("@ symbol present")
    if extracted_features.get('sslfinal_state', -1) == 1:
        warning_signs.append("No HTTPS")
    if extracted_features.get('having_sub_domain', -1) == 1:
        warning_signs.append("Many subdomains")
    if extracted_features.get('shortining_service', -1) == 1:
        warning_signs.append("URL shortener")
    if extracted_features.get('prefix_suffix', -1) == 1:
        warning_signs.append("Hyphen in domain")
    if extracted_features.get('https_token', -1) == 1:
        warning_signs.append("HTTPS in domain name")
    if extracted_features.get('abnormal_url', -1) == 1:
        warning_signs.append("Suspicious patterns detected")
    
    # Risk level
    if risk_score >= 8:
        risk_level = "HIGH"
    elif risk_score >= 5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # 🔥 FIX: Convert ALL numpy/non-serializable types to Python native types
    return {
        "prediction": str(result),
        "confidence": round(float(confidence), 4),
        "phishing_probability": round(float(probabilities[1]), 4),
        "legitimate_probability": round(float(probabilities[0]), 4),
        "type": "Website URL Analysis",
        "url": str(url),
        "domain": str(domain),
        "has_https": bool(has_https),
        "url_length": int(url_length),
        "risk_score": int(risk_score),
        "risk_level": str(risk_level),
        "warning_signs": [str(w) for w in warning_signs],
        "active_risk_factors": [str(k) for k, v in risk_factors.items() if int(v) > 0]
    }
# --------------------------------------------------
# 🔥 RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Phishing Detection API - Hybrid System")
    print("="*60)
    print(f"✓ Email Model: {'Loaded' if email_model else 'Not Found'}")
    print(f"✓ Website Model: {'Loaded' if website_model else 'Not Found'}")
    print("="*60)
    print("📡 Server running at: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
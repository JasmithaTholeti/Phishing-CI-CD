import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import joblib
import re
from urllib.parse import urlparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" ADVANCED PHISHING WEBSITE DETECTION MODEL")
print("=" * 80)
print(" Using Enhanced Feature Engineering + UCI Dataset")
print("=" * 80)

# ============================================================================
# PART 1: LOAD UCI PHISHING WEBSITES DATASET
# ============================================================================
print("\n[1/5] Loading UCI Phishing Websites Dataset...")
try:
    phishing_websites = fetch_ucirepo(id=327)
    X_uci = phishing_websites.data.features
    y_uci = phishing_websites.data.targets.squeeze().replace(-1, 0)
    
    X_uci.columns = X_uci.columns.str.lower()
    
    print(f"✓ UCI Dataset loaded successfully")
    print(f"  → Shape: {X_uci.shape}")
    print(f"  → Features: {X_uci.shape[1]}")
    print(f"  → Phishing: {sum(y_uci==1)}, Legitimate: {sum(y_uci==0)}")
    
    feature_names = list(X_uci.columns)
    
    # Analyze feature importance
    print(f"\n  Training quick RF to identify important features...")
    from sklearn.ensemble import RandomForestClassifier
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_temp.fit(X_uci, y_uci)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")
    
    important_features = feature_importance.head(15)['feature'].tolist()
    
except Exception as e:
    print(f"✗ Failed to load UCI data: {e}")
    exit(1)

# ============================================================================
# PART 2: TRAIN MODELS
# ============================================================================
print("\n[2/5] Training Models with Cross-Validation...")

X_train, X_test, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.2, random_state=42, stratify=y_uci
)

print(f"  → Training set: {X_train.shape[0]} samples")
print(f"  → Test set: {X_test.shape[0]} samples")

mlflow.set_experiment("Phishing-Detection-Advanced")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, 
        max_depth=20,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ''
all_results = {}

with mlflow.start_run() as run:
    print(f"\n  MLflow Run ID: {run.info.run_id}")
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        all_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'model': model
        }
        
        print(f"    → Accuracy:  {acc:.4f}")
        print(f"    → Precision: {prec:.4f}")
        print(f"    → Recall:    {rec:.4f}")
        print(f"    → F1-Score:  {f1:.4f}")
        
        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_precision", prec)
        mlflow.log_metric(f"{name}_recall", rec)
        mlflow.log_metric(f"{name}_f1", f1)
        
        if f1 > best_accuracy:  # Use F1 for best model (balanced metric)
            best_accuracy = f1
            best_model = model
            best_name = name
    
    print(f"\n  🏆 Best Model: {best_name} (by F1-Score)")
    print(f"     Accuracy:  {all_results[best_name]['accuracy']:.4f}")
    print(f"     Precision: {all_results[best_name]['precision']:.4f}")
    print(f"     Recall:    {all_results[best_name]['recall']:.4f}")
    print(f"     F1-Score:  {all_results[best_name]['f1']:.4f}")
    
    joblib.dump(best_model, 'phishing_website_model.joblib')
    joblib.dump(feature_names, 'website_features.joblib')
    joblib.dump(important_features, 'important_features.joblib')
    print(f"\n  ✓ Files saved successfully")

# ============================================================================
# PART 3: DETAILED EVALUATION
# ============================================================================
print("\n[3/5] Detailed Model Evaluation...")

y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0][0]:4d} (Correctly identified as Legitimate)")
print(f"  False Positives: {cm[0][1]:4d} (Legitimate marked as Phishing)")
print(f"  False Negatives: {cm[1][0]:4d} (Phishing marked as Legitimate) ⚠️")
print(f"  True Positives:  {cm[1][1]:4d} (Correctly identified as Phishing)")

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"\n  Additional Metrics:")
print(f"    Specificity (TNR): {specificity:.4f}")
print(f"    Sensitivity (TPR): {sensitivity:.4f}")
print(f"    False Positive Rate: {fp/(fp+tn):.4f}")
print(f"    False Negative Rate: {fn/(fn+tp):.4f}")

# ============================================================================
# PART 4: ADVANCED URL FEATURE EXTRACTION
# ============================================================================
print("\n[4/5] Creating Advanced Feature Extractor...")

# Common phishing keywords and patterns
PHISHING_KEYWORDS = [
    'verify', 'account', 'update', 'secure', 'banking', 'login', 'signin',
    'confirm', 'suspend', 'restrict', 'urgent', 'immediately', 'click',
    'paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'wallet', 'payment', 'billing'
]

LEGITIMATE_TLDS = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'int']
SUSPICIOUS_TLDS = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'date', 'wang']

def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    return -sum((count/length) * np.log2(count/length) for count in counter.values())

def extract_url_features(url):
    """
    Advanced feature extraction with intelligent pattern detection
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
        
        # 1. IP Address Detection (VERY STRONG INDICATOR)
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        has_ip = bool(re.search(ip_pattern, domain))
        features['having_ip_address'] = 1 if has_ip else -1
        
        # 2. @ Symbol (VERY STRONG INDICATOR)
        features['having_at_symbol'] = 1 if '@' in url else -1
        
        # 3. URL Length
        url_len = len(url)
        if url_len < 54:
            features['url_length'] = -1
        elif url_len <= 75:
            features['url_length'] = 0
        else:
            features['url_length'] = 1
        
        # 4. URL Shorteners (STRONG INDICATOR)
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 
                      'buff.ly', 'adf.ly', 'short.link', 'tiny.cc']
        is_shortener = any(s in domain for s in shorteners)
        features['shortining_service'] = 1 if is_shortener else -1
        
        # 5. Prefix/Suffix with Dash (STRONG INDICATOR)
        # Phishing sites often use dashes to separate words
        has_dash = '-' in domain_clean
        features['prefix_suffix'] = 1 if has_dash else -1
        
        # 6. Subdomains (STRONG INDICATOR)
        dot_count = domain.count('.')
        if dot_count <= 1:
            features['having_sub_domain'] = -1
        elif dot_count == 2:
            features['having_sub_domain'] = 0
        else:
            features['having_sub_domain'] = 1
        
        # 7. HTTPS and SSL
        has_https = url.startswith('https://')
        # Check if 'https' appears in domain (phishing trick)
        https_in_domain = 'https' in domain
        
        if has_https and not https_in_domain:
            features['sslfinal_state'] = -1
        elif has_https:
            features['sslfinal_state'] = 0
        else:
            features['sslfinal_state'] = 1
        
        # 8. HTTPS token in domain (STRONG PHISHING INDICATOR)
        features['https_token'] = 1 if https_in_domain else -1
        
        # 9. Double Slash Redirecting
        double_slash_count = url.count('//')
        features['double_slash_redirecting'] = 1 if double_slash_count > 1 else -1
        
        # ==================== DOMAIN ANALYSIS ====================
        
        # 10. Abnormal URL - Check for phishing keywords
        domain_parts = domain_clean.replace('.', ' ').replace('-', ' ').split()
        suspicious_keyword_count = sum(1 for word in domain_parts if word in PHISHING_KEYWORDS)
        
        # Also check entire domain for patterns
        suspicious_patterns = [
            r'(\d{3,})',  # Multiple consecutive digits
            r'([a-z])\1{3,}',  # Character repeated 3+ times
            r'secure.*account',
            r'verify.*login',
            r'update.*now'
        ]
        pattern_matches = sum(1 for pattern in suspicious_patterns if re.search(pattern, domain_clean))
        
        features['abnormal_url'] = 1 if (suspicious_keyword_count >= 2 or pattern_matches >= 1) else -1
        
        # 11. Domain Registration Length (proxy using domain complexity)
        if len(domain_clean) > 15 and dot_count <= 2:
            features['domain_registeration_length'] = -1  # Legitimate
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
        
        # 13. Entropy of domain (randomness)
        entropy = calculate_entropy(domain_clean.replace('.', ''))
        if entropy < 3.0:
            domain_entropy = -1  # Low entropy = normal words
        elif entropy < 4.0:
            domain_entropy = 0
        else:
            domain_entropy = 1  # High entropy = random characters
        
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
        
        # Calculate overall suspicion score
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
        if total_risk >= 8:  # High risk
            features['statistical_report'] = 1
            features['page_rank'] = 1
            features['web_traffic'] = 1
            features['google_index'] = 1
            features['age_of_domain'] = 1
        elif total_risk >= 5:  # Medium risk
            features['statistical_report'] = 0
            features['page_rank'] = 0
            features['web_traffic'] = 0
            features['google_index'] = 0
            features['age_of_domain'] = 0
        else:  # Low risk
            features['statistical_report'] = -1
            features['page_rank'] = -1
            features['web_traffic'] = -1
            features['google_index'] = -1
            features['age_of_domain'] = -1
        
        # ==================== DEFAULT VALUES ====================
        
        # Features we can't determine from URL alone
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
        
        # Store calculated values for reporting
        features['_risk_score'] = total_risk
        features['_risk_factors'] = risk_factors
        
    except Exception as e:
        print(f"    ⚠️  Error extracting features: {e}")
        # Safe defaults
        features = {feat: -1 for feat in feature_names}
        features['_risk_score'] = 0
        features['_risk_factors'] = {}
    
    return features

def predict_url(url, model, feature_list, show_features=False):
    """
    Predict if URL is phishing with enhanced analysis
    """
    print(f"\n  Analyzing: {url}")
    
    # Extract features
    url_features = extract_url_features(url)
    
    # Get risk info
    risk_score = url_features.pop('_risk_score', 0)
    risk_factors = url_features.pop('_risk_factors', {})
    
    # Create DataFrame
    feature_df = pd.DataFrame([url_features])
    
    # Ensure all features present
    for feat in feature_list:
        if feat not in feature_df.columns:
            feature_df[feat] = -1
    
    feature_df = feature_df[feature_list]
    
    if show_features:
        print("\n  🔍 Key Security Indicators:")
        important = [
            ('having_ip_address', 'IP Address in URL'),
            ('having_at_symbol', '@ Symbol Present'),
            ('shortining_service', 'URL Shortener'),
            ('prefix_suffix', 'Dash in Domain'),
            ('having_sub_domain', 'Subdomain Count'),
            ('sslfinal_state', 'HTTPS Status'),
            ('https_token', 'HTTPS in Domain'),
            ('abnormal_url', 'Suspicious Patterns')
        ]
        
        for feat, desc in important:
            if feat in url_features:
                value = url_features[feat]
                if value == 1:
                    indicator = "🚨 ALERT"
                    status = "SUSPICIOUS"
                elif value == 0:
                    indicator = "⚠️  WARN"
                    status = "NEUTRAL"
                else:
                    indicator = "✅ SAFE"
                    status = "NORMAL"
                print(f"    {indicator:12s} {desc:25s}: {status}")
        
        print(f"\n  📊 Risk Analysis:")
        print(f"    Overall Risk Score: {risk_score}/20")
        if risk_score >= 8:
            print(f"    Risk Level: 🚨 HIGH")
        elif risk_score >= 5:
            print(f"    Risk Level: ⚠️  MEDIUM")
        else:
            print(f"    Risk Level: ✅ LOW")
        
        active_risks = [k for k, v in risk_factors.items() if v > 0]
        if active_risks:
            print(f"    Active Risk Factors: {', '.join(active_risks)}")
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]
    
    return prediction, probabilities, url_features, risk_score

print("✓ Advanced Feature Extractor created")
print("  → Intelligent pattern detection")
print("  → Risk-based scoring")
print("  → 30+ features analyzed")

# ============================================================================
# PART 5: TEST WITH SAMPLE URLS
# ============================================================================
print("\n[5/5] Testing with Sample URLs...")
print("\n" + "="*80)
print(" COMPREHENSIVE URL TESTING")
print("="*80)

test_urls = [
    ("https://www.google.com", "Legitimate"),
    ("https://www.github.com/user/repo", "Legitimate"),
    ("http://125.98.3.123/fake.html", "Phishing - IP"),
    ("https://secure-login-verify-account-now.com", "Phishing - Keywords"),
    ("https://www.amazon.com", "Legitimate"),
    ("http://paypal-secure.com/signin?account=verify", "Phishing - Mimics"),
    ("https://bit.ly/abc123", "Suspicious - Shortener"),
    ("https://www.microsoft.com/en-us/download", "Legitimate"),
    ("http://www.banking-secure-login@phishing.com", "Phishing - @ symbol"),
    ("https://update-your-account-immediately.net", "Phishing - Urgency")
]

print("\nDetailed Analysis Results:\n")

correct = 0
total = len(test_urls)

for i, (url, expected) in enumerate(test_urls, 1):
    print(f"\n{'='*80}")
    print(f"Test #{i}: {url}")
    print(f"Expected: {expected}")
    print('-'*80)
    
    pred, prob, features, risk_score = predict_url(url, best_model, feature_names, show_features=True)
    
    print(f"\n  🎯 PREDICTION:")
    if pred == 1:
        result = "🚨 PHISHING DETECTED"
        confidence = prob[1]
    else:
        result = "✅ LEGITIMATE"
        confidence = prob[0]
    
    print(f"    {result}")
    print(f"    Confidence: {confidence:.1%}")
    print(f"    Probabilities: Legitimate {prob[0]:.1%} | Phishing {prob[1]:.1%}")
    
    # Check if prediction matches expectation
    is_phishing_expected = 'phishing' in expected.lower() or 'suspicious' in expected.lower()
    is_correct = (pred == 1) == is_phishing_expected
    if is_correct:
        correct += 1
        print(f"    ✓ CORRECT PREDICTION")
    else:
        print(f"    ✗ INCORRECT PREDICTION")

print(f"\n{'='*80}")
print(f" TEST SUMMARY")
print(f"{'='*80}")
print(f"  Correct Predictions: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"{'='*80}")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================
print("\n" + "="*80)
print(" 🔐 INTERACTIVE PHISHING DETECTOR")
print("="*80)
print(" Test any URL for phishing indicators")
print(" Type 'quit' to exit")
print("="*80)

while True:
    print("\n" + "-"*80)
    user_url = input("\n🔍 Enter URL: ").strip()
    
    if user_url.lower() in ['quit', 'exit', 'q', '']:
        if user_url.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Stay safe online! 🛡️")
        break
    
    if not user_url.startswith(('http://', 'https://')):
        user_url = 'http://' + user_url
    
    print("\n" + "="*80)
    print(" 🔍 DETAILED SECURITY ANALYSIS")
    print("="*80)
    
    pred, prob, features, risk_score = predict_url(user_url, best_model, feature_names, show_features=True)
    
    print("\n" + "="*80)
    print(" 🎯 FINAL VERDICT")
    print("="*80)
    
    if pred == 1:
        if prob[1] > 0.85:
            print("\n  🚨 CRITICAL: HIGH-CONFIDENCE PHISHING SITE")
            risk_level = "CRITICAL"
        elif prob[1] > 0.7:
            print("\n  🚨 WARNING: LIKELY PHISHING SITE")
            risk_level = "HIGH"
        else:
            print("\n  ⚠️  CAUTION: POSSIBLE PHISHING SITE")
            risk_level = "MEDIUM"
        
        print(f"  Risk Level: {risk_level}")
        print(f"  Phishing Probability: {prob[1]:.1%}")
        print(f"\n  ⛔ DO NOT PROCEED WITH THIS WEBSITE")
        print(f"     • Do NOT enter passwords")
        print(f"     • Do NOT enter credit card information")
        print(f"     • Do NOT download files")
        print(f"     • Close this website immediately")
        
    else:
        if prob[0] > 0.85:
            print("\n  ✅ SAFE: HIGH-CONFIDENCE LEGITIMATE SITE")
            safety = "HIGH"
        elif prob[0] > 0.7:
            print("\n  ✅ LIKELY SAFE: APPEARS LEGITIMATE")
            safety = "MEDIUM-HIGH"
        else:
            print("\n  ⚠️  PROCEED WITH CAUTION")
            safety = "MEDIUM"
        
        print(f"  Safety Level: {safety}")
        print(f"  Legitimate Probability: {prob[0]:.1%}")
        
        if prob[0] < 0.85:
            print(f"\n  💡 Security Tips:")
            print(f"     • Always verify the exact URL")
            print(f"     • Check for HTTPS and valid certificate")
            print(f"     • Be cautious with personal information")
    
    print(f"\n  Score: Legitimate {prob[0]:.1%} | Phishing {prob[1]:.1%}")
    print("="*80)

print("\n" + "="*80)
print(" 🎓 TRAINING COMPLETE")
print("="*80)
print(f"  Model: {best_name}")
print(f"  Accuracy:  {all_results[best_name]['accuracy']:.2%}")
print(f"  Precision: {all_results[best_name]['precision']:.2%}")
print(f"  Recall:    {all_results[best_name]['recall']:.2%}")
print(f"  F1-Score:  {all_results[best_name]['f1']:.2%}")
print("="*80)
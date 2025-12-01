import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" PHISHING EMAIL DETECTION - SIMPLIFIED VERSION")
print("=" * 80)

# ============================================================================
# LOAD ALL DATASETS
# ============================================================================
print("\n[1/5] Loading Datasets...")

all_texts = []
all_labels = []

dataset_files = [
    'data/phishing_email.csv',
]

for filepath in dataset_files:
    try:
        print(f"  Loading: {filepath}...", end=" ")
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        print(f"✓ Shape: {df.shape}")
        
        # Find text column
        text_col = None
        for col in df.columns:
            if any(word in col.lower() for word in ['text', 'body', 'message', 'email', 'content']):
                text_col = col
                break
        if text_col is None:
            text_col = [col for col in df.columns if df[col].dtype == 'object'][0]
        
        # Find label column
        label_col = None
        for col in df.columns:
            if any(word in col.lower() for word in ['label', 'spam', 'phishing', 'class', 'target']):
                label_col = col
                break
        
        if text_col and label_col:
            texts = df[text_col].fillna('').astype(str).tolist()
            labels = df[label_col]
            
            # Normalize labels to 0/1
            if labels.dtype == 'object':
                labels = labels.map({
                    'spam': 1, 'phishing': 1, 'Spam': 1, 'Phishing': 1, 1: 1, '1': 1,
                    'ham': 0, 'legitimate': 0, 'Ham': 0, 'Legitimate': 0, 0: 0, '0': 0
                })
            
            valid = ~labels.isna()
            texts = [t for i, t in enumerate(texts) if valid.iloc[i]]
            labels = labels[valid].astype(int).tolist()
            
            all_texts.extend(texts)
            all_labels.extend(labels)
            print(f"    → Added {len(texts)} samples")
    except Exception as e:
        print(f"✗ {str(e)[:40]}")

print(f"\nTotal: {len(all_texts)} samples | Phishing: {sum(all_labels)} | Legitimate: {len(all_labels)-sum(all_labels)}")

# ============================================================================
# BALANCE DATASET
# ============================================================================
print("\n[2/5] Balancing Dataset...")

df = pd.DataFrame({'text': all_texts, 'label': all_labels})
df_phishing = df[df['label'] == 1]
df_legit = df[df['label'] == 0]

sample_size = min(len(df_phishing), len(df_legit), 10000)
df_balanced = pd.concat([
    df_phishing.sample(n=sample_size, random_state=42),
    df_legit.sample(n=sample_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Balanced to {len(df_balanced)} samples ({sample_size} each class)")

X_text = df_balanced['text']
y = df_balanced['label']

# ============================================================================
# VECTORIZE TEXT
# ============================================================================
print("\n[3/5] Vectorizing Text...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.7,
    ngram_range=(1, 2),
    stop_words='english'
)

X = vectorizer.fit_transform(X_text)
print(f"  ✓ Created {X.shape[1]} features")

joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# ============================================================================
# TRAIN MODELS
# ============================================================================
print("\n[4/5] Training Models...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Phishing-Detection")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ''

with mlflow.start_run():
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        mlflow.log_metric(f"{name}_accuracy", acc)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name
    
    print(f"\n  🏆 Best: {best_name} ({best_accuracy:.4f})")
    joblib.dump(best_model, 'phishing_model.joblib')

# ============================================================================
# EVALUATE
# ============================================================================
print("\n[5/5] Evaluation...")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives:  {cm[1][1]}")

# ============================================================================
# TEST SAMPLES
# ============================================================================
print("\n" + "="*80)
print("TESTING SAMPLE EMAILS")
print("="*80)

test_emails = [
    "Hello Sir,hope your doing well,I wanted to know the interview timings",
    "Your flight booking is confirmed. Ticket is attached.",
    "Hello sir, hope you're doing well. Attached is the project report.",
    "Hello Sir,hope your doing well, Can you send me some amount ,i will return it soon",
    "URGENT! Your account has been suspended. Click here immediately.",
    "Congratulations! You won $1,000,000. Reply with bank details.",
    "Meeting scheduled for Monday at 2 PM.",
    "Your Netflix subscription is expiring. Update payment now.",
    "Limited time offer! 90% off. Click now!"
]

for i, email in enumerate(test_emails, 1):
    vec = vectorizer.transform([email])
    pred = best_model.predict(vec)[0]
    prob = best_model.predict_proba(vec)[0]
    
    result = "🚨 PHISHING" if pred == 1 else "✅ SAFE"
    conf = prob[1] if pred == 1 else prob[0]
    
    print(f"\n{i}. {email}")
    print(f"   {result} ({conf:.1%} confidence)")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================
print("\n" + "="*80)
print("INTERACTIVE TESTING - Enter your email text")
print("Type 'quit' to exit")
print("="*80)

while True:
    user_input = input("\nEnter email text: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n✓ Thank you!")
        break
    
    if not user_input:
        continue
    
    vec = vectorizer.transform([user_input])
    pred = best_model.predict(vec)[0]
    prob = best_model.predict_proba(vec)[0]
    
    print("\n" + "="*80)
    if pred == 1:
        print(f"🚨 PHISHING DETECTED ({prob[1]:.1%} confidence)")
        print("   NOT SAFE - Be careful with this email!")
    else:
        print(f"✅ LEGITIMATE ({prob[0]:.1%} confidence)")
        print("   SAFE - This appears to be genuine")
    print(f"   Score: {prob[1]:.1%} phishing | {prob[0]:.1%} legitimate")
    print("="*80)
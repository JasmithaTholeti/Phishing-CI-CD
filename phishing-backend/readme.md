# Hybrid Phishing Detection Backend

This directory contains the server-side logic for the Phishing Detection System. It features a **modular architecture** that separates Natural Language Processing (NLP) for emails and heuristic analysis for website URLs, exposed via a **FastAPI** server.

---

## 📂 File Structure

### 🧠 Core Logic & Training

The training pipeline has been decoupled into specialized scripts for better maintainability:

* **`train_emails.py`**: Handles the NLP pipeline. It loads the email dataset, vectorizes text using TF-IDF, trains the email classifier, and saves `tfidf_vectorizer.joblib` and `phishing_model.joblib`.
* **`train_websites.py`**: Handles the structural analysis. It processes the UCI website dataset, selects critical features, trains the website classifier, and saves `phishing_website_model.joblib` and `website_features.joblib`.
* **`main.py`**: The production API server. It loads artifacts from both training pipelines to perform hybrid inference on incoming requests.
* **`check_drift.py`**: The monitoring utility. It performs statistical analysis on `prediction_log.csv` to detect model drift and triggers automated retraining via GitHub Actions.
* **`kafka.py`**: A simulation utility acting as a Kafka Consumer to demonstrate real-time stream processing and monitoring capabilities.

### ⚙️ Configuration & Infrastructure

* **`Dockerfile`**: Defines the environment for containerizing the application (`Python 3.10-slim` base).
* **`requirements.txt`**: Lists all Python dependencies (pinned versions recommended for production).
* **`prediction_log.csv`**: *(Generated at runtime)* Stores live prediction data for drift analysis.

### 📦 Model Artifacts

* `phishing_model.joblib`: The trained classifier for Email Text.
* `phishing_website_model.joblib`: The trained classifier for Website Features.
* `tfidf_vectorizer.joblib`: The NLP vectorizer for converting text to machine-readable numbers.
* `website_features.joblib`: List of specific features used by the website model.
* `important_features.joblib`: Stores feature importance rankings for explainability.

---

## 🚀 Setup & Execution Guide

### Option 1: Running via Docker (Recommended)

This method ensures a consistent environment and avoids dependency conflicts.

1.  **Build the Image**

    ```bash
    docker build -t phishing-backend .
    ```

2.  **Run the Container**
    > **Note:** The `GITHUB_TOKEN` is required for the automated retraining feature to work.

    ```bash
    docker run -p 8000:8000 -e GITHUB_TOKEN="your_personal_access_token" phishing-backend
    ```

### Option 2: Running Locally

1.  **Create & Activate Virtual Environment**

    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Models** (If artifacts are missing)
    *You must run both scripts to generate all necessary `.joblib` files.*

    ```bash
    python train_emails.py
    python train_websites.py
    ```

4.  **Verify Model Generation**

    ```bash
    ls -lh *.joblib
    echo "✓ All model files generated successfully"
    ```

5.  **Start the API Server**

    ```bash
    uvicorn main:app --reload
    ```

    The API will be available at `http://localhost:8000`.

---

## 📡 API Endpoints

### `POST /predict`

The endpoint accepts a JSON payload containing either email text **OR** website features.

| Payload Example (Email Mode) | Payload Example (Website Mode) |
| :--- | :--- |
| ```json { "email_text": "URGENT: Verify your account immediately at [http://bit.ly/fake-link](http://bit.ly/fake-link)" } ``` | ```json { "website_features": { "having_ip_address": 1, "sslfinal_state": -1, "url_length": 1, ... (other 27 features) } } ``` |

---

## 🔍 Monitoring & Stream Simulation

### Drift Detection

To check the health of the model and detect potential drift:

```bash
python check_drift.py
```

## 🔍 Monitoring & Stream Simulation

### Kafka Stream Simulation

To demonstrate real-time data processing (**Producer/Consumer** pattern) without full infrastructure overhead:

```bash
python kafka.py
```

## 🔄 CI/CD Pipeline Configuration

The project uses **GitHub Actions** for Continuous Integration and Deployment. To enable the Docker build and push steps, you must configure **Repository Secrets**.

### 1. Required Secrets

Go to **Settings** > **Secrets and variables** > **Actions** > **New repository secret** and add:

* `DOCKER_USERNAME`: Your Docker Hub username.
* `DOCKER_PASSWORD`: Your Docker Hub Access Token (recommended) or password.

### 2. Workflow Definition

The pipeline (defined in `.github/workflows/main_pipeline.yml`) automatically handles the build process using these secrets.

> **Note:** In your own fork, replace `your-docker-username` with your actual username in the Docker build step.

```yaml
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
          
      - name: Build and Push Docker image
        run: |
          cd ..
          # Replace 'your-docker-username' with your actual Docker Hub ID
          docker build -t your-docker-username/phishing-backend:latest ./phishing-backend
          docker push your-docker-username/phishing-backend:latest

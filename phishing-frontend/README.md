# Phishing Detection Frontend

This directory contains the client-side application for the Phishing Detection System. It is a single-page application (SPA) built with **React** that provides an interactive dashboard for the hybrid machine learning model.

---

## 📖 Overview

The frontend serves as the **presentation layer**, enabling users to interact with the backend API through two distinct modes:

* **Email Text Analysis:** Accepts raw email content for NLP-based phishing detection.
* **Website Feature Analysis:** Provides a togglable interface to configure 30 heuristic features (based on the UCI dataset) for structural analysis.

---

## 🛠️ Technical Stack

| Category | Technology | Details |
| :--- | :--- | :--- |
| **Framework** | React.js | v18+ |
| **Styling** | Tailwind CSS | Via CDN integration in `public/index.html` |
| **Icons** | Lucide React | |
| **State Management** | React Hooks | `useState` |
| **Network** | Native Fetch API | |

---

## 📋 Prerequisites

Ensure the following are installed on your development machine:

* **Node.js:** Version 18.0.0 or higher
* **npm:** Version 9.0.0 or higher (usually bundled with Node.js)

---

## 🚀 Installation & Setup

1.  **Navigate** to the frontend directory:

    ```bash
    cd phishing-frontend
    ```

2.  **Install** the required dependencies:
    *This installs React, Lucide icons, and other necessary packages listed in `package.json`.*

    ```bash
    npm install
    ```

---

## 🏃‍♂️ Execution

To start the development server:

```bash
npm start
```

## 🔌 Configuration

### Backend Connection

The application is hardcoded to communicate with the backend service at **`http://localhost:8000`**.

* **Endpoint Used:** `POST /predict`
* **Error Handling:** If the backend is unreachable, the UI will display a specific error message ("Failed to connect to backend").

> **Note:** Ensure the Backend (FastAPI) is running via Docker or locally on port **8000** before performing an analysis.

---

## 🖥️ Usage Guide

### 1. Email Analysis Mode

1.  Select the **"Email Text Analysis"** tab.
2.  Paste the raw text of a suspicious email into the text area.
3.  Click **Analyze Threat** to send the payload to the NLP model.

### 2. Website Feature Analysis Mode

1.  Select the **"Website Feature Analysis"** tab.
2.  Manually configure the 30 feature dropdowns (`1 = Legitimate`, `0 = Suspicious`, `-1 = Phishing`).
3.  **Presets:** Use the **"Load Phishing"** or **"Load Safe"** buttons to auto-populate the form with known patterns for rapid testing without manual entry.

---

## ❓ Troubleshooting

### Issue: "Failed to connect to backend"

* **Cause:** The backend API is not running or is not accessible on port 8000.
* **Resolution:** Ensure the Docker container or local Python process is active.
    * **Docker:** `docker run -p 8000:8000 phishing-backend`
    * **Local:** `uvicorn main:app --reload`

### Issue: Styling looks broken / Plain HTML

* **Cause:** Tailwind CSS failed to load.
* **Resolution:** Check your internet connection.The project uses a CDN link in `public/index.html` to load Tailwind styles.

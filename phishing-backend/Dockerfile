# 1. Base Image: Use a lightweight version of Python 3.10
FROM python:3.10-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy the requirements file first
# We do this BEFORE copying the rest of the code to use Docker's "Layer Caching".
# If you change your code but not your requirements, Docker skips this slow step!
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
# This includes main.py, check_drift.py, and all .joblib model files
COPY . .

# 6. Expose Port 8000
# This tells Docker that the container listens on this port
EXPOSE 8000

# 7. Start the Application
# --host 0.0.0.0 is CRITICAL for Docker (allows outside connections)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
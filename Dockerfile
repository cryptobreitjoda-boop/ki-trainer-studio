FROM python:3.11-slim

WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Requirements zuerst für Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn jupyter ipykernel joblib

# Code
COPY . .

# Jupyter Kernel
RUN python -m ipykernel install --user --name=ki-trainer

# Ports
EXPOSE 8000 8888 6006

# Volumes für Daten & Modelle
VOLUME ["/app/data", "/app/uploads", "/app/models"]

CMD ["bash", "-c", "echo '=== KI Trainer Studio ===' && echo 'API: http://localhost:8000/docs' && echo 'Jupyter: http://localhost:8888' && echo 'TensorBoard: http://localhost:6006' && uvicorn api_server:app --host 0.0.0.0 --port 8000"]

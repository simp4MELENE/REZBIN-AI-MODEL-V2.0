FROM python:3.11-slim

WORKDIR /app

COPY requirements-build.txt .
RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir torch==2.7.0+cpu torchvision==0.22.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-build.txt && \
    apt-get purge -y gcc && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app

# install all requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all models in to docker build
COPY models/ ./models/

# copy entire repo and ignore the ones specified in .dockerignore
COPY . .

# replace `--host` with `--reload` if using for development
CMD ["uvicorn", "app:app", "--host", "--port", "8000"]
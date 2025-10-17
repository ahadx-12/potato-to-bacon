# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and web UI
COPY src/ ./src/
COPY web/ ./web/

# Ensure the web directory exists
RUN ls -R /app/web

EXPOSE 8000

# Start FastAPI (Railway sets $PORT automatically)
CMD ["uvicorn", "potatobacon.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

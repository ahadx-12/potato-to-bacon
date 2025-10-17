# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy pyproject.toml first for dependency caching
COPY pyproject.toml .

# Install project dependencies using PEP 517 (pyproject.toml)
RUN pip install --no-cache-dir .

# Copy application source code and UI
COPY src/ ./src/
COPY web/ ./web/

# Verify that the UI files were copied
RUN ls -R /app/web

# Expose default app port
EXPOSE 8000

# Start FastAPI (Railway automatically sets $PORT)
CMD ["uvicorn", "potatobacon.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

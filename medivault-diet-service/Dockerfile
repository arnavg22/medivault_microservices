# ================================
# MediVault Diet Service — Dockerfile
# Optimised for Railway (low-RAM, no sentence-transformers)
# ================================

FROM python:3.11-slim-bookworm

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Switch to non-root user
USER appuser

# Railway injects PORT env var
ENV PORT=5001
EXPOSE 5001

# Start FastAPI via gunicorn — $PORT is set by Railway at runtime
CMD exec gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120

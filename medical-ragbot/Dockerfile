# ================================
# MediVault RAG Bot — Dockerfile
# Optimised for Render deployment
# ================================

FROM python:3.11-slim-bookworm

# Install system-level dependencies for OCR
# tesseract-ocr   → Tesseract OCR engine (replaces Windows install)
# poppler-utils   → pdftoppm for pdf2image (replaces Windows Poppler)
# libgl1           → required by some Pillow/OpenCV builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories (PDFs ingested at runtime, not baked in)
RUN mkdir -p data/raw_pdfs data/processed_text

# Pre-download the fastembed model into the image so startup is instant
# and there is no extra RAM spike from downloading at runtime.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))" 

# Expose port (Render uses $PORT env var)
EXPOSE 8000

# Start FastAPI via uvicorn
# Render sets $PORT automatically — default to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

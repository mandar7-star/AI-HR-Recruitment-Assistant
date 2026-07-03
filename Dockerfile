FROM python:3.11-slim

# Create a non-root user (required convention for HF Spaces)
RUN useradd -m -u 1000 user

# System deps needed by chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the backend code
COPY --chown=user ./agents.py ./models.py ./main.py ./tasks.py ./tools.py ./

# Redirect all caches to /tmp — HF Spaces containers are read-only
# outside of /tmp and the user's home directory
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface \
    CHROMA_TELEMETRY=false \
    PYTHONUNBUFFERED=1

USER user

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
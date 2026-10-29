# Sightector API + dashboard (CPU inference).
#   docker build -t sightector .
#   docker run -p 5000:5000 sightector      -> http://127.0.0.1:5000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SIGHTECTOR_MODEL=transformer \
    PORT=5000

WORKDIR /app

# CPU-only PyTorch (~200 MB instead of ~2.5 GB for CUDA), then the rest of the
# runtime dependencies. The CUDA index line in requirements.txt is skipped.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
 && grep -v -e "^--extra-index-url" -e "^torch" requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

COPY sightector/ sightector/
COPY server/ server/
# Whatever models exist locally: the baseline is in git; transformer/bilstm are
# added when they have been trained. The server falls back transformer -> bilstm -> baseline.
COPY artifacts/models/ artifacts/models/

RUN useradd --create-home --uid 1000 sightector && chown -R sightector /app
USER sightector

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=4 --call server.app:create_app"]

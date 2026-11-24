FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
# --require-hashes: refuse any package whose hash is not in the lock file (supply-chain protection)
RUN pip install --require-hashes -r requirements.txt
COPY app ./app
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]

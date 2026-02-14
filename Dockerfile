# Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# (optional) tiny utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

# ----------------------------
# Image variant A: expects model to be mounted at runtime
# ----------------------------
FROM base AS mount-model
CMD ["python", "-m", "src"]

# ----------------------------
# Image variant B: bakes model into image
# ----------------------------
FROM base AS with-model
COPY models ./models
CMD ["python", "-m", "src"]
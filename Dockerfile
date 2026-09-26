# ---- Stage 1: build the React frontend ----
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + ffmpeg ----
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/tmp/outputs \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./backend/
RUN cd backend && uv sync --frozen --no-dev

COPY backend/ ./backend/
COPY data/brands ./data/brands
COPY data/ads ./data/ads
COPY --from=web /web/dist ./frontend/dist

# Render injects $PORT; default for local `docker run`
EXPOSE 10000
# --proxy-headers: behind Render's TLS proxy, so manifest URLs come out as https://
CMD ["sh", "-c", "backend/.venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*'"]

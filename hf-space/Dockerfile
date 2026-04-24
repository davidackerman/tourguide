# Tourguide cloud analysis backend — runs on Hugging Face Spaces (Docker SDK).
#
# This container exposes a small FastAPI app that the tourguide web-app can
# optionally call when a user's analysis is too heavy for Pyodide's 4 GB
# WASM ceiling. Everything is additive — the web-app works fine without this.

FROM python:3.11-slim

# Build deps needed by a few of the scientific Python wheels (cc3d has a
# native extension; scipy/skimage ship manylinux wheels so no compiler is
# usually required there).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. HF Spaces expects the app to run as uid 1000 with a
# writable $HOME. We write session data under /tmp at runtime.
RUN useradd -m -u 1000 analyst
USER analyst
ENV HOME=/home/analyst \
    PATH=/home/analyst/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=analyst:analyst requirements.txt /app/
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=analyst:analyst . /app/

EXPOSE 7860

# HF Spaces routes public HTTPS traffic to whatever port the container
# exposes via app_port in README.md (7860 by convention).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]

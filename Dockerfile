FROM python:3.13.0-slim-bookworm@sha256:0de818129b26ed8f46fd772f540c80e277b67a28229531a1ba0fdacfaed19bcb
ARG MODEL_SOURCE=models/final/pipeline.joblib
ARG MODEL_FILENAME=pipeline.joblib
ARG MODEL_DIGEST=e355fb092ca5c6d3f8498369001a803c9afc399b82bdd379ee61b902809af372

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONTZPATH="" \
    TZ=America/Mexico_City \
    MODEL_PATH=/app/models/${MODEL_FILENAME} \
    MODEL_SHA256=${MODEL_DIGEST}

WORKDIR /app
COPY requirements-linux-py313.txt ./
RUN python -m pip install --only-binary=:all: pip==26.3 \
    && python -m pip install --only-binary=:all: --no-deps -r requirements-linux-py313.txt \
    && python -m pip check \
    && groupadd --gid 10001 mibici \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin mibici

COPY --chown=10001:10001 mibici_demand/__init__.py mibici_demand/api.py mibici_demand/config.py mibici_demand/features.py mibici_demand/release.py ./mibici_demand/
COPY --chown=10001:10001 mibici_demand/modeling/__init__.py mibici_demand/modeling/pipeline.py ./mibici_demand/modeling/
COPY --chown=10001:10001 ${MODEL_SOURCE} ./models/${MODEL_FILENAME}
COPY --chown=10001:10001 references/model_release.json ./references/model_release.json
COPY --chown=10001:10001 examples/predict_request.json examples/predict_request_final.json ./examples/

USER 10001:10001
RUN python -c "from pathlib import Path; from mibici_demand.api import load_model; import os; print(load_model(Path(os.environ['MODEL_PATH'])).info.model_version)"

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "mibici_demand.api:app", "--host", "0.0.0.0", "--port", "8000"]

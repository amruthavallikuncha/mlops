FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY artifacts/models ./artifacts/models

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && test -f /app/artifacts/models/heart_disease_pipeline.joblib \
    && test -f /app/artifacts/models/model_metadata.json

RUN groupadd --gid 10001 appgroup \
    && useradd --create-home --uid 10001 --gid 10001 appuser \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
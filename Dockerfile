FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-container.txt .
RUN python -m pip install --no-cache-dir -r requirements-container.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/artifacts /app/data/interim /app/models \
        /app/great_expectations/uncommitted /mlflow /var/log/evcharge \
    && chown -R app:app /app/artifacts /app/data/interim /app/models \
        /app/great_expectations /mlflow /var/log/evcharge

COPY configs ./configs
COPY src ./src
COPY data/sample ./data/sample
COPY scripts ./scripts

USER app

EXPOSE 8000 5000

CMD ["uvicorn", "evcharge.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]

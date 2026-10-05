FROM python:3.12-slim

WORKDIR /app 

RUN useradd --create-home appuser && chown appuser:appuser /app


RUN pip install uv 

COPY --chown=appuser:appuser pyproject.toml .

COPY --chown=appuser:appuser uv.lock .

COPY --chown=appuser:appuser app/ app/

USER appuser

RUN uv sync --frozen --no-dev


EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


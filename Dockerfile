FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[rag,ui]"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/agentic_rag/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Only what the app needs at runtime
COPY .streamlit/ .streamlit/
COPY app/ app/
COPY src/features.py src/features.py
COPY models/ebm.joblib models/random_forest.joblib models/model_meta.json models/
COPY data_processed/paddy_dataset.csv data_processed/paddy_dataset.csv

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

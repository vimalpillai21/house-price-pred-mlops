FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY api/ ./api/
COPY models/ ./models/
COPY helper_functions/ ./helper_functions/
COPY configs/ ./configs/
COPY main.py .

RUN mkdir -p /mlflow/artifacts
RUN useradd -m appuser && chown -R appuser /app
RUN chown -R appuser /mlflow

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app","--host","0.0.0.0","--port","8000"]
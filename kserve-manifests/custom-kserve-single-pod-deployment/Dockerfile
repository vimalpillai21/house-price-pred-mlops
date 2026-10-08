FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.py /app/
ENTRYPOINT ["python", "model.py"]

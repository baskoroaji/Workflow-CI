FROM python:3.12-slim


RUN pip install --no-cache-dir mlflow scikit-learn pandas uvicorn fastapi prometheus_client


COPY model /app/model

WORKDIR /app

COPY inference.py /app/inference.py


EXPOSE 8080
EXPOSE 8000


ENTRYPOINT ["mlflow", "models", "serve", \
            "--model-uri", "model", \
            "--no-conda", \
            "--host", "0.0.0.0", \
            "--port", "8080"]
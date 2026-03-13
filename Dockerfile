FROM python:3.11

RUN pip install mlflow psycopg2-binary boto3

WORKDIR /mlflow

EXPOSE 5050

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5050 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow/artifacts
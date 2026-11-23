# MLflow server cho model registry: metadata trong SQLite, file model trong thư mục (hoặc Cloudflare R2 qua API S3).
# Image chính thức của MLflow không có boto3 (cần khi chuyển sang S3/R2), nên tự build.
FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==3.18.0" "boto3==1.44.9"

EXPOSE 5000

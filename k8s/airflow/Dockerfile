FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
USER airflow

# Install project dependencies (including boto3 for MinIO S3 and pyarrow for parquet)
COPY --chown=airflow:root pyproject.toml /opt/airflow/
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost optuna category-encoders \
    mlflow prometheus-client python-dotenv kaggle pydantic httpx joblib psycopg2-binary \
    boto3 pyarrow

# Copy DAGs and source code
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root src/ /opt/airflow/src/

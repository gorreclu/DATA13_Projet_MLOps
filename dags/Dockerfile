FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
USER airflow

# Install Python dependencies needed by DAGs
RUN pip install --no-cache-dir \
    boto3 pyarrow pandas numpy scikit-learn xgboost optuna \
    category-encoders mlflow kaggle joblib python-dotenv dill

# Copy DAGs and source code
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root src/ /opt/airflow/src/

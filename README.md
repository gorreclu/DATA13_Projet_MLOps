# House Price Prediction -- MLOps Project

> **DATA713 -- Mastere Specialise IA Expert Data / MLOps**
> Telecom Paris -- Projet POC en une journee

---

## Overview

End-to-end MLOps platform for predicting house prices in King County, Washington.
Trained on 21,613 historical transactions using **XGBoost** with **Optuna** hyperparameter
optimization (10 trials), served through a **FastAPI** REST API and a **Gradio** web interface,
orchestrated by **Apache Airflow**, tracked with **MLflow**, stored on **MinIO**
(S3-compatible object storage), monitored via **Prometheus + Grafana**, and deployable
on **Kubernetes**.

---

## Architecture

```
                        +-------------------+
                        |   Gradio WebApp   |  :7860
                        +--------+----------+
                                 |
                        +--------v----------+
                        |   FastAPI (API)   |  :8000  --> /metrics (Prometheus)
                        +--------+----------+
                                 |
                  +--------------+--------------+
                  |                             |
        +---------v---------+       +-----------v----------+
        | MLflow (EXTERNAL) |       | XGBoost Model        |
        | 89.58.44.97:5000  |       | (loaded from MLflow) |
        +--------+----------+       +----------------------+
                 |
        +--------v----------+       +--------------------+
        |   PostgreSQL       |      |   MinIO (S3)       |  :9000 / :9001
        |   :5432            |      |   (training data)  |
        +--------------------+      +--------------------+

        +--------------------+      +--------------------+
        |   Apache Airflow   |      |   Prometheus       |  :9090
        |   :8080            |      +--------+-----------+
        +--------------------+               |
                                    +--------v----------+
                                    |   Grafana          |  :3000
                                    +--------------------+
```

> **Note**: MLflow is an **external** service hosted at `http://89.58.44.97:5000`.
> It is not deployed inside the Kubernetes cluster.

---

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| ML Model         | XGBoost + Optuna (10 trials)        |
| Experiment Tracking | MLflow (external, 89.58.44.97:5000) |
| Object Storage   | MinIO (S3-compatible, :9000)        |
| API              | FastAPI + Uvicorn                   |
| WebApp           | Gradio                              |
| Orchestration    | Apache Airflow                      |
| Database         | PostgreSQL 16                       |
| Monitoring       | Prometheus + Grafana                |
| Alerting         | Gmail SMTP                          |
| CI/CD            | GitHub Actions                      |
| Containerization | Docker + Docker Compose             |
| Deployment       | Kubernetes (Minikube)               |
| Package Manager  | uv                                  |
| Language         | Python 3.11                         |

---

## Project Structure

```
.
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                  #   Lint (Ruff) + Test (pytest) + Docker build check
│   └── cd.yml                  #   Build/push images to GHCR + deploy to K8s
├── dags/                       # Airflow DAGs
│   ├── Dockerfile              #   Custom Airflow image (DAGs + src/ + deps)
│   ├── training_dag.py         #   Train → Compare → Promote/Reject → Alert
│   └── data_pipeline_dag.py    #   Data freshness check → Download → Trigger training
├── k8s/                        # Kubernetes manifests
│   ├── api/                    #   API Deployment + Service + ConfigMap
│   ├── webapp/                 #   WebApp Deployment + Service
│   ├── airflow/                #   Airflow Deployment + Service + PVC
│   ├── minio/                  #   MinIO Deployment + Service + PVC
│   ├── postgresql/             #   PostgreSQL Deployment + Service + PVC
│   ├── monitoring/             #   Prometheus + Grafana (deployments, dashboards)
│   └── secrets.yaml.example    #   Template for the app-secrets Secret
├── grafana/provisioning/       # Grafana provisioning (Docker Compose)
│   ├── datasources/            #   Prometheus datasource config
│   └── dashboards/             #   Dashboard JSON + provider config
├── src/
│   ├── config.py               # Centralized configuration (env vars + defaults)
│   ├── data/
│   │   ├── download.py         #   Kaggle dataset download
│   │   ├── preprocess.py       #   Cleaning, feature engineering, encoding
│   │   └── s3_storage.py       #   MinIO S3 upload/download helpers
│   ├── model/
│   │   ├── cli.py              #   CLI entry points (train, train-quick)
│   │   ├── train.py            #   XGBoost + Optuna + MLflow training loop
│   │   ├── evaluate.py         #   Metrics (RMSE, MAE, R2, MAPE) + model comparison
│   │   └── predict.py          #   Model loading from MLflow + inference
│   ├── api/
│   │   ├── app.py              #   FastAPI application + Prometheus metrics
│   │   ├── schemas.py          #   Pydantic request/response schemas
│   │   └── Dockerfile
│   ├── webapp/
│   │   ├── app.py              #   Gradio web interface
│   │   └── Dockerfile
│   └── utils/
│       └── alerting.py         #   Gmail SMTP email alerts
├── tests/
│   ├── test_preprocess.py      # Preprocessing unit tests
│   ├── test_model.py           # Model training/prediction tests
│   └── test_api.py             # API endpoint tests
├── docker-compose.yml          # Full dev environment (8 services)
├── pyproject.toml              # Dependencies & CLI scripts (uv)
├── init-db.sql                 # PostgreSQL init (create mlflow + airflow DBs)
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose
- A Kaggle account (for dataset download)
- (Optional) Minikube for K8s deployment

### 1. Setup

```bash
git clone git@github.com:gorreclu/DATA13_Projet_MLOps.git && cd DATA13_Projet_MLOps

# Copy and fill environment variables
cp .env.example .env
# Edit .env: add your Kaggle credentials, SMTP settings, etc.

# Install Python dependencies
uv sync --all-extras
```

### 2. Download Data

```bash
uv run download
```

### 3. Train the Model

```bash
# Full training with Optuna optimization (10 trials)
uv run train

# Quick training without optimization (default XGBoost params)
uv run train-quick
```

### 4. Run with Docker Compose (recommended)

```bash
docker compose up -d --build

# Services:
#   API:        http://localhost:8000/docs
#   WebApp:     http://localhost:7860
#   Airflow:    http://localhost:8080   (admin / admin)
#   MinIO:      http://localhost:9001   (minioadmin / minioadmin)
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000   (admin / admin)
```

### 5. Run Locally (without Docker)

```bash
# Terminal 1: API
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: WebApp
uv run python -m src.webapp.app
```

### 6. Deploy to Kubernetes (Minikube)

```bash
# Start Minikube
minikube start

# Build images inside Minikube
eval $(minikube docker-env)
docker build -f src/api/Dockerfile -t house-price-api:latest .
docker build -f src/webapp/Dockerfile -t house-price-webapp:latest .
docker build -f dags/Dockerfile -t airflow-custom:latest .

# Create secrets (edit the file with your Base64-encoded credentials first)
cp k8s/secrets.yaml.example k8s/secrets.yaml
# echo -n "your_value" | base64   # to encode each value
kubectl apply -f k8s/secrets.yaml

# Deploy all manifests
kubectl apply -f k8s/ --recursive

# Check status
kubectl get pods

# Access services (macOS: NodePort IPs are not reachable, use port-forward)
kubectl port-forward svc/api 8000:8000 &
kubectl port-forward svc/webapp 7860:7860 &
kubectl port-forward svc/airflow 8080:8080 &
kubectl port-forward svc/minio 9001:9001 &
kubectl port-forward svc/prometheus 9090:9090 &
kubectl port-forward svc/grafana 3000:3000 &
```

| Service    | URL                          | Credentials         |
|------------|------------------------------|---------------------|
| API docs   | http://localhost:8000/docs   | --                  |
| WebApp     | http://localhost:7860        | --                  |
| Airflow    | http://localhost:8080        | admin / admin       |
| MinIO      | http://localhost:9001        | minioadmin / minioadmin |
| Prometheus | http://localhost:9090        | --                  |
| Grafana    | http://localhost:3000        | admin / admin       |

```bash
# Tear down
kubectl delete -f k8s/ --recursive
minikube stop
```

### 7. Tests & Linting

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## Environment Variables

### How secrets are managed

| Context | Mechanism | File |
|---------|-----------|------|
| **Local dev** (no Docker) | `.env` loaded by `python-dotenv` at import time | `.env` |
| **Docker Compose** | `.env` auto-loaded by Compose + `x-common-env` YAML anchor | `docker-compose.yml` |
| **Kubernetes** | `ConfigMap` (non-sensitive) + `Secret` (credentials) | `k8s/*/deployment.yaml` + `k8s/secrets.yaml` |
| **GitHub Actions CI** | Env vars in workflow YAML (non-sensitive only) | `.github/workflows/ci.yml` |
| **GitHub Actions CD** | `secrets.GITHUB_TOKEN` + `secrets.KUBECONFIG` | `.github/workflows/cd.yml` |

### Variable reference

| Variable | Sensitive? | Dev source | Prod source |
|----------|-----------|------------|-------------|
| `KAGGLE_USERNAME` | Yes | `.env` | K8s Secret `app-secrets` |
| `KAGGLE_KEY` | Yes | `.env` | K8s Secret `app-secrets` |
| `SMTP_EMAIL` | Yes | `.env` | K8s Secret `app-secrets` |
| `SMTP_PASSWORD` | Yes | `.env` | K8s Secret `app-secrets` |
| `POSTGRES_USER` | Yes | `.env` | K8s Secret `app-secrets` |
| `POSTGRES_PASSWORD` | Yes | `.env` | K8s Secret `app-secrets` |
| `MINIO_ROOT_USER` | Yes | `.env` | K8s Secret `app-secrets` |
| `MINIO_ROOT_PASSWORD` | Yes | `.env` | K8s Secret `app-secrets` |
| `MLFLOW_TRACKING_URI` | No | `.env` | K8s ConfigMap |
| `MLFLOW_EXPERIMENT_NAME` | No | `.env` | K8s ConfigMap |
| `MINIO_ENDPOINT` | No | `.env` | K8s ConfigMap |
| `MINIO_BUCKET_DATA` | No | `.env` | K8s ConfigMap |
| `API_HOST` / `API_PORT` | No | `.env` | K8s ConfigMap |

### Best practices applied

1. **Never commit secrets**: `.env` and `k8s/secrets.yaml` are in `.gitignore`
2. **Templates provided**: `.env.example` and `k8s/secrets.yaml.example` are committed
3. **Single source of truth**: `src/config.py` centralizes all env var lookups with sensible defaults
4. **Separation of concerns**: ConfigMaps for config, Secrets for credentials (K8s)
5. **Kaggle API key**: set `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env` (or place `kaggle.json` in `~/.kaggle/`)
6. **Gmail SMTP**: use an [App Password](https://myaccount.google.com/apppasswords), never your real password

---

## ML Pipeline

### Preprocessing

1. **Cleaning** -- Parse dates, drop `id`, remove 33-bedroom outlier, impute `waterfront` NaN
2. **Feature Engineering** -- `house_age`, `is_renovated`, `years_since_renovation`, `has_basement`,
   area ratios (living vs neighbors, lot vs neighbors)
3. **Train/Test Split** -- 80/20 (before encoding to prevent leakage)
4. **Target Encoding** -- Zipcode via `category_encoders.TargetEncoder` (fit on train only)
5. **Target Transformation** -- `log1p(price)` for training, `expm1()` at inference

### Continuous Training (Airflow)

| DAG | Schedule | Description |
|-----|----------|-------------|
| `data_pipeline` | Weekly | Check data freshness on MinIO, download from Kaggle, validate, trigger training |
| `training_pipeline` | Triggered | Download → Preprocess → Train (Optuna + MLflow) → **Compare with production model** → Promote or reject → Alert |

The training pipeline implements **model comparison**: after training, the new model is
evaluated against the current production model on the test set. It is promoted only if
it achieves a lower RMSE. Otherwise the previous model stays in production.

### Monitoring

- **Prometheus** scrapes `/metrics` on the API every 10s
- **Grafana** dashboard displays:
  - Model KPIs: RMSE, MAE, R2, MAPE
  - API metrics: request count, error count, latency percentiles (p50/p95/p99)
  - Training info: total runs, data rows processed

---

## CI/CD

- **CI** (push/PR to `main` or `dev`): Ruff lint + format check → pytest with coverage → Docker build test
- **CD** (on tag `v*`): Build & push images to GHCR → Deploy to Kubernetes

---

## Dataset

- **Source**: [King County House Sales](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Size**: 21,613 transactions, 21 features
- **Period**: May 2014 -- May 2015
- **Target**: `price` (USD)

---

## Team

DATA713 MLOps Project -- Mastere Specialise IA Expert Data, Telecom Paris

## License

This project is for educational purposes as part of the DATA713 course.

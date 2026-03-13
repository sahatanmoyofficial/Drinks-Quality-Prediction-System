# 🍷 Drinks Quality Prediction System (DQPS)

> **Predicting Drink Quality from Physicochemical Properties with Machine Learning**
>
> An end-to-end MLOps system that takes 11 physicochemical measurements of a drink sample and predicts its **quality score** — with a modular 5-stage pipeline, schema-validated data, a Flask web interface, and automated CI/CD to AWS.

---

<div align="center">

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Latest-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Model-ElasticNet-orange)](https://scikit-learn.org/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR-orange?logo=amazonaws)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want a visual overview first?** Browse the full presentation before diving into the code.

👉 **[View the Project Presentation (PPTX)](./DQPS_Project_Slides.pptx)**

The deck covers: business problem → architecture → pipeline stages → model results → web UI → deployment — in 12 slides.

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Data & Features](#6-data--features) |
| 7 | [ML Pipeline — Step by Step](#7-ml-pipeline--step-by-step) |
| 8 | [Model Performance](#8-model-performance) |
| 9 | [Web Application](#9-web-application) |
| 10 | [How to Replicate — Full Setup Guide](#10-how-to-replicate--full-setup-guide) |
| 11 | [Running the Application](#11-running-the-application) |
| 12 | [CI/CD & Cloud Deployment](#12-cicd--cloud-deployment) |
| 13 | [Business Applications & Other Industries](#13-business-applications--other-industries) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Business Problem

### What problem are we solving?

Drink quality — particularly for wine — is traditionally assessed by human tasters, a process that is expensive, slow, subjective, and impossible to scale. Physicochemical laboratory tests (measuring acidity, sugar content, pH, alcohol levels, etc.) already exist as part of standard quality control, but there is no automated system to translate these measurements into a predicted quality score.

This creates real operational pain:

- 🏭 **Producers** can't rapidly screen batches before bottling — costly defects pass to market
- 📊 **Quality managers** have no objective, data-driven baseline to compare against sensory scores
- 💸 **Cost of recalls** from poorly rated products reaching consumers is significant
- 🔬 **Researchers and food scientists** lack a quick prediction tool for formulation experiments

### What does DQPS answer?

> *"Given the 11 measurable physicochemical properties of a drink sample, what quality score (0–10) should it receive?"*

### Objectives

1. Build a regression model that predicts drink quality from physicochemical features
2. Validate incoming data against a strict schema before processing
3. Serve predictions via a user-friendly Flask web interface — no code required
4. Structure the codebase in clean, modular components following production ML standards
5. Deploy automatically to AWS EC2 via Docker and GitHub Actions CI/CD

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | Red wine physicochemical properties (1,599 samples) |
| **Source URL** | `https://github.com/entbappy/Branching-tutorial/raw/master/Drinks-data.zip` |
| **Target** | `quality` — integer score from 0 to 10 |
| **Task** | Regression (predict numeric quality score) |
| **Train / Test Split** | 75 / 25 (default `sklearn` split) |
| **Model** | ElasticNet (Linear Regression with L1 + L2 regularisation) |
| **Hyperparameters** | `alpha=0.2`, `l1_ratio=0.1` (from `params.yaml`) |
| **Metrics** | RMSE: 0.6898 · MAE: 0.5536 · R²: 0.2527 |
| **Pipeline** | 5 modular stages (Ingest → Validate → Transform → Train → Evaluate) |
| **API Framework** | Flask on port 8080 |
| **Frontend** | HTML/CSS/Jinja2 web form (Bootstrap 5) |
| **Deployment** | Docker → AWS ECR → AWS EC2 |
| **CI/CD** | GitHub Actions (self-hosted runner on EC2) |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.8 | Core language for entire ML pipeline and API |
| **ML Model** | Scikit-learn `ElasticNet` | Linear regression with combined L1 + L2 regularisation |
| **Data Processing** | Pandas, NumPy | Data loading, splitting, feature/target separation |
| **Model Serialisation** | joblib | Save/load `model.joblib` |
| **Config Management** | PyYAML + python-box | Read `config.yaml`, `params.yaml`, `schema.yaml` as dot-accessible objects |
| **Schema Enforcement** | `ensure` library + `schema.yaml` | Type-checks all input columns before processing |
| **Web Framework** | Flask + Flask-CORS | Serves the prediction UI and `/train` trigger endpoint |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Jinja2 | Two-page web app (form + results) |
| **Containerisation** | Docker (`python:3.8.5-slim-buster`) | Packages app for consistent deployment |
| **Cloud Compute** | AWS EC2 (Ubuntu) | Hosts Flask application as Docker container |
| **Container Registry** | AWS ECR | Stores Docker images pushed by GitHub Actions |
| **CI/CD** | GitHub Actions | Build → push to ECR → deploy to EC2 on every push |
| **Logging** | Python `logging` | Structured logs to `logs/running_logs.log` |
| **Utilities** | `python-box`, `ensure`, `tqdm` | Config access, type annotations, progress bars |
| **Package Management** | `setuptools` (`setup.py`) | Installs `mlProject` as an editable package from `src/` |

---

## 4. High-Level Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│                                                                 │
│   [ GitHub ZIP URL ]  ──►  [ artifacts/data_ingestion/ ]       │
│                              data.zip → Drinks-data.csv         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE LAYER                             │
│                                                                 │
│  [Stage 1]    [Stage 2]    [Stage 3]    [Stage 4]   [Stage 5]  │
│  Ingestion → Validation → Transform →  Trainer  → Evaluation   │
│     │            │            │           │           │         │
│     ▼            ▼            ▼           ▼           ▼         │
│  data.zip   status.txt   train.csv   model.joblib metrics.json  │
│                         test.csv                                │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                             │
│                                                                 │
│  [ Flask app.py ]  port 8080                                    │
│     GET  /         → index.html (prediction form)              │
│     GET  /train    → triggers main.py pipeline                  │
│     POST /predict  → loads model.joblib, returns quality score  │
│        │                                                        │
│  [ Browser ]  fills 11-field HTML form → sees quality score     │
│        │                                                        │
│  [ Docker Container ]  ◄──  [ AWS EC2 ]                        │
│                                  ▲                              │
│  [ GitHub Push ] → [ Actions ] → [ ECR ] → [ EC2 Deploy ]      │
└─────────────────────────────────────────────────────────────────┘
```

### Data & Config Flow Summary

| # | Stage | Input | Output |
|---|-------|-------|--------|
| 1 | **Ingestion** | GitHub ZIP URL (`config.yaml`) | `artifacts/data_ingestion/Drinks-data.csv` |
| 2 | **Validation** | `Drinks-data.csv` + `schema.yaml` | `artifacts/data_validation/status.txt` (True/False) |
| 3 | **Transformation** | `Drinks-data.csv` | `artifacts/data_transformation/train.csv` + `test.csv` |
| 4 | **Model Training** | `train.csv` + `params.yaml` | `artifacts/model_trainer/model.joblib` |
| 5 | **Evaluation** | `test.csv` + `model.joblib` | `artifacts/model_evaluation/metrics.json` |
| 6 | **Serving** | 11 form inputs → `model.joblib` | Quality score rendered in `results.html` |

---

## 5. Repository Structure

```
Drinks-Quality-Prediction-System/
│
├── src/mlProject/                      # Core Python package (installed as -e .)
│   ├── __init__.py                     # Logger initialisation
│   ├── components/                     # Stage implementations
│   │   ├── data_ingestion.py           # Downloads ZIP from URL, extracts CSV
│   │   ├── data_validation.py          # Column-by-column schema check
│   │   ├── data_transformation.py      # train_test_split (75/25)
│   │   ├── model_trainer.py            # ElasticNet fit + joblib save
│   │   └── model_evaluation.py         # RMSE, MAE, R2 → metrics.json
│   ├── config/
│   │   └── configuration.py            # ConfigurationManager — reads all 3 YAMLs
│   ├── constants/
│   │   └── __init__.py                 # Paths to config.yaml, params.yaml, schema.yaml
│   ├── entity/
│   │   └── config_entity.py            # Frozen dataclasses for each stage config
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py               # PredictionPipeline — loads model, predicts
│   └── utils/
│       └── common.py                   # read_yaml, create_directories, save_json, get_size
│
├── config/
│   └── config.yaml                     # File paths and data source URL
├── params.yaml                         # ElasticNet hyperparameters
├── schema.yaml                         # Expected column names and data types
│
├── artifacts/                          # Pipeline outputs (auto-created, gitignored)
│   ├── data_ingestion/                 # data.zip + Drinks-data.csv
│   ├── data_validation/                # status.txt
│   ├── data_transformation/            # train.csv + test.csv
│   ├── model_trainer/                  # model.joblib
│   └── model_evaluation/              # metrics.json
│
├── templates/
│   ├── index.html                      # 11-field prediction form (Bootstrap 5)
│   └── results.html                    # Displays predicted quality score
│
├── static/
│   ├── assets/                         # favicon, background images
│   ├── css/styles.css                  # Bootstrap theme CSS
│   └── js/scripts.js                   # Bootstrap JS
│
├── research/                           # Jupyter notebooks (exploratory work)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── Expriement.ipynb
│   └── trials.ipynb
│
├── data/
│   └── Drinks-data.csv                 # Local copy of raw dataset (1,599 rows)
│
├── main.py                             # Runs all 5 pipeline stages sequentially
├── app.py                              # Flask application (GET /, GET /train, POST /predict)
├── Dockerfile                          # FROM python:3.8.5-slim-buster, CMD python3 app.py
├── requirements.txt                    # All Python dependencies
├── setup.py                            # Package: mlProject
├── template.py                         # Scaffold script to create project folder structure
└── logs/running_logs.log               # Pipeline execution logs
```

---

## 6. Data & Features

### Dataset

**Name:** Red Wine Quality Dataset
**Rows:** 1,599 samples
**Source:** Downloaded from GitHub as a ZIP file at pipeline runtime

### Input Features (11 physicochemical properties)

| Feature | Type | Description |
|---------|------|-------------|
| `fixed acidity` | float64 | Non-volatile acids (tartaric, malic, etc.) — affects taste stability |
| `volatile acidity` | float64 | Acetic acid content — high values produce vinegar taste |
| `citric acid` | float64 | Adds freshness and flavour; found in small quantities |
| `residual sugar` | float64 | Sugar remaining after fermentation — affects sweetness |
| `chlorides` | float64 | Salt content — influences saltiness perception |
| `free sulfur dioxide` | float64 | Free SO₂ — acts as antimicrobial and antioxidant |
| `total sulfur dioxide` | float64 | Total SO₂ (free + bound) — regulated for health reasons |
| `density` | float64 | Closely linked to alcohol and sugar content |
| `pH` | float64 | Describes acidity/basicity (most wines: 3–4) |
| `sulphates` | float64 | Wine additive that contributes to SO₂ levels |
| `alcohol` | float64 | Percentage of alcohol by volume |

### Target Variable

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `quality` | int64 | 0–10 | Sensory quality score from expert tasters |

### Data Validation

Before any transformation, `data_validation.py` checks every column in the loaded CSV against `schema.yaml`. If any column is missing or unexpected, `status.txt` is written as `False` and the pipeline can be halted. The current validation result is:

```
Validation status: True
```

### Data Transformation

The transformation step in this project is intentionally minimal — the dataset is already clean:

```python
train, test = train_test_split(data)   # 75% train, 25% test (sklearn default)
```

No scaling, encoding, or imputation is applied. This is noted as an improvement opportunity (see Section 14).

---

## 7. ML Pipeline — Step by Step

The pipeline is triggered by running `main.py`, which executes all 5 stages sequentially. Each stage is a self-contained class that reads its config from `ConfigurationManager`.

The pattern across every stage is identical:

```python
config = ConfigurationManager()
stage_config = config.get_<stage>_config()
component = <Component>(config=stage_config)
component.<run_method>()
```

---

### Stage 1 — Data Ingestion

**Component:** `src/mlProject/components/data_ingestion.py`
**Config entity:** `DataIngestionConfig`

1. Reads `source_URL` from `config.yaml`
2. If `data.zip` does not already exist locally, downloads it via `urllib.request.urlretrieve`
3. Extracts the ZIP to `artifacts/data_ingestion/` → produces `Drinks-data.csv`
4. Skips download if file already exists (logs the existing file size)

---

### Stage 2 — Data Validation

**Component:** `src/mlProject/components/data_validation.py`
**Config entity:** `DataValidationConfig`

1. Reads `Drinks-data.csv` into a DataFrame
2. Reads expected columns from `schema.yaml` (`COLUMNS` section)
3. Checks every column in the CSV against the schema
4. Writes `True` to `status.txt` if all columns match, `False` if any column is unexpected
5. Returns the validation boolean for downstream use

---

### Stage 3 — Data Transformation

**Component:** `src/mlProject/components/data_transformation.py`
**Config entity:** `DataTransformationConfig`

1. Reads `Drinks-data.csv`
2. Splits 75/25 using `sklearn.model_selection.train_test_split` (no `random_state` set)
3. Saves `artifacts/data_transformation/train.csv` and `test.csv`

---

### Stage 4 — Model Training

**Component:** `src/mlProject/components/model_trainer.py`
**Config entity:** `ModelTrainerConfig`

1. Reads `train.csv` from `artifacts/data_transformation/`
2. Separates features (11 columns) from target (`quality`)
3. Reads `alpha` and `l1_ratio` from `params.yaml`
4. Trains `ElasticNet(alpha=0.2, l1_ratio=0.1, random_state=42)`
5. Saves trained model as `artifacts/model_trainer/model.joblib` via `joblib.dump`

**About ElasticNet:** ElasticNet combines L1 (Lasso) and L2 (Ridge) regularisation. `l1_ratio=0.1` means 10% L1 + 90% L2 — predominantly Ridge behaviour, which shrinks all coefficients but keeps all features. `alpha=0.2` controls the overall regularisation strength.

---

### Stage 5 — Model Evaluation

**Component:** `src/mlProject/components/model_evaluation.py`
**Config entity:** `ModelEvaluationConfig`

1. Loads `model.joblib` and `test.csv`
2. Generates predictions on the test set
3. Computes three metrics:

| Metric | Formula | Result |
|--------|---------|--------|
| **RMSE** | `√(mean((actual - predicted)²))` | **0.6898** |
| **MAE** | `mean(|actual - predicted|)` | **0.5536** |
| **R²** | Explained variance ratio | **0.2527** |

4. Saves all three metrics to `artifacts/model_evaluation/metrics.json`

---

## 8. Model Performance

### Current Results

```json
{
    "rmse": 0.6897990431838804,
    "mae":  0.5535615348113513,
    "r2":   0.2526662622691097
}
```

### Plain-English Interpretation

| Metric | Value | What It Means |
|--------|-------|---------------|
| **RMSE 0.69** | The model's predictions are off by ~0.69 quality points on average (penalises large errors more) |
| **MAE 0.55** | The typical prediction error is about half a quality point — e.g. predicting 5.4 when the true score is 6.0 |
| **R² 0.25** | The model explains ~25% of the variance in quality scores |

### Honest Assessment

An R² of 0.25 is modest — the model is a useful baseline but leaves significant unexplained variance. This is expected because:

- Wine quality scores from human experts are inherently subjective and noisy
- ElasticNet is a linear model; quality likely has non-linear relationships with some features
- No feature engineering or scaling has been applied

The RMSE/MAE values (~0.55–0.69 on a 0–10 scale) are practically meaningful — predictions are typically within one quality grade of the true score. More powerful models (see Section 14) would substantially improve these numbers.

### Model Configuration

```yaml
# params.yaml
ElasticNet:
  alpha: 0.2      # Overall regularisation strength
  l1_ratio: 0.1   # 10% L1 (Lasso) + 90% L2 (Ridge)
```

---

## 9. Web Application

The Flask web app (`app.py`) provides a browser-based interface — no code or API knowledge required.

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Renders `index.html` — the 11-field prediction form |
| `GET` | `/train` | Triggers `python main.py` — runs the full pipeline, re-trains the model |
| `POST` | `/predict` | Reads 11 form fields, calls `PredictionPipeline.predict()`, renders `results.html` |

### How the Prediction Form Works

1. User fills in 11 physicochemical values (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
2. Clicks **Predict**
3. `app.py` reads each field as `float`, assembles a `numpy` array of shape `(1, 11)`
4. Passes it to `PredictionPipeline.predict(data)` which loads `model.joblib` and returns the score
5. Result rendered in `results.html` as `{{ prediction }}`

### Example Postman / curl Test

```bash
curl -X POST http://localhost:8080/predict \
  -d "fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0.0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11.0&total_sulfur_dioxide=34.0&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4"
```

---

## 10. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.8+
- Git
- Conda or `venv`
- AWS account (for deployment only)
- Docker Desktop (optional, for local container testing)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Drinks-Quality-Prediction-System.git
cd Drinks-Quality-Prediction-System
```

---

### Step 2 — Set Up Python Environment

```bash
# Conda (recommended)
conda create -n mlproj python=3.8 -y
conda activate mlproj

# Or venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# Installs the mlProject package in editable mode (-e .)
# Verify: python -c "from mlProject import logger; print('Package OK')"
```

---

### Step 4 — Review Configuration (Optional)

All configuration is in three YAML files — no hardcoding anywhere in the source:

```bash
# config/config.yaml  — file paths and data source URL
# params.yaml         — ElasticNet hyperparameters (alpha, l1_ratio)
# schema.yaml         — expected column names and types
```

To change hyperparameters, edit `params.yaml` and re-run training:

```yaml
ElasticNet:
  alpha: 0.5      # Try higher for more regularisation
  l1_ratio: 0.5   # 50/50 L1 and L2 blend
```

---

### Step 5 — Run the Full Pipeline

```bash
python main.py
```

This executes all 5 stages in order. After completion, verify outputs:

```bash
ls artifacts/data_ingestion/    # data.zip, Drinks-data.csv
ls artifacts/data_validation/   # status.txt → should contain "Validation status: True"
ls artifacts/data_transformation/ # train.csv, test.csv
ls artifacts/model_trainer/     # model.joblib
cat artifacts/model_evaluation/metrics.json  # RMSE, MAE, R2
```

---

### Step 6 — Launch the Web App

```bash
python app.py
# App starts at http://localhost:8080
```

Navigate to `http://localhost:8080` — fill the form with physicochemical values and click **Predict**.

To re-train via the browser, visit `http://localhost:8080/train`.

---

## 11. Running the Application

### Local Run

```bash
python app.py
# Runs on http://0.0.0.0:8080
```

### Docker Run

```bash
# Build
docker build -t dqps-app:latest .

# Run
docker run -d -p 8080:8080 dqps-app:latest

# Test
curl http://localhost:8080/
```

### Retrain via Browser

Visit `http://localhost:8080/train` — this calls `os.system("python main.py")` on the server, re-running all 5 pipeline stages and overwriting `model.joblib` with a freshly trained model.

---

## 12. CI/CD & Cloud Deployment

Every push to `main` triggers the GitHub Actions workflow:

```
Developer ──► git push origin main
                     │
              GitHub Actions triggered
                     │
        ┌────────────┴────────────────────────────────────┐
        │   Job 1: CI                │   Job 2: CD        │
        │   (ubuntu-latest)          │   (self-hosted EC2) │
        │                            │                     │
        │  1. Checkout               │  1. Checkout        │
        │  2. Configure AWS creds    │  2. Login to ECR    │
        │  3. Login to ECR           │  3. docker run -d   │
        │  4. docker build           │     -p 8080:8080    │
        │  5. docker push → ECR      │     (env secrets)   │
        └────────────────────────────┴─────────────────────┘
                                              │
                                    Flask App live at
                                    http://<EC2-IP>:8080
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository URI |

### EC2 Setup for Self-Hosted Runner

```bash
# Install Docker on EC2
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker

# Register self-hosted runner
# GitHub → Settings → Actions → Runners → New self-hosted runner → Linux
# Follow the provided commands on your EC2 instance
```

### AWS Infrastructure Overview

| Resource | Purpose |
|----------|---------|
| EC2 instance | Hosts Flask app as Docker container |
| ECR repository | Stores Docker images |
| IAM user | GitHub Actions deployment credentials |
| Security Group | Open port 8080 for Flask app |

---

## 13. Business Applications & Other Industries

### Primary Use Case — Drinks Quality Control

| User | Value Delivered |
|------|----------------|
| **Wine producers** | Pre-bottling quality screening from lab measurements — avoid releasing poor batches |
| **Quality managers** | Objective data-driven baseline alongside subjective taster scores |
| **Food scientists** | Rapid quality prediction during formulation experiments |
| **Regulatory bodies** | Automated screening at scale for quality certification programmes |
| **Distributors** | Verify quality claims from suppliers before purchase |

### Adjacent Industries (Same Regression Pattern)

| Industry | Analogous Problem | Key Adaptations |
|----------|------------------|----------------|
| **Brewing / Beer** | Predict beer quality from fermentation chemistry | Retrain on beer physicochemical dataset |
| **Olive oil** | Predict oil grade (extra virgin, virgin, lampante) | Reclassify target as ordinal categories |
| **Dairy** | Milk quality prediction from fat/protein/acidity | Add microbiological features |
| **Water treatment** | Potability prediction from dissolved minerals | Binary target (safe / unsafe) |
| **Pharmaceutical** | Tablet hardness/dissolution from formulation properties | Strict FDA-compliant validation pipeline |
| **Semiconductor manufacturing** | Wafer quality from process parameters | Higher-dimensional feature space |
| **Agriculture** | Crop yield prediction from soil chemistry | Seasonal and geo features needed |

---

## 14. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Switch to ensemble models** | 🔴 High | Random Forest or Gradient Boosting (XGBoost/LightGBM) would likely push R² from 0.25 to 0.55+ |
| **Feature scaling** | 🔴 High | Add `StandardScaler` or `RobustScaler` before ElasticNet — linear models are sensitive to feature magnitudes |
| **Hyperparameter tuning** | 🔴 High | Use `GridSearchCV` or `Optuna` to find optimal `alpha` and `l1_ratio` |
| **Feature engineering** | 🟡 Medium | Add interaction terms (e.g. `alcohol × pH`), log-transform skewed features |
| **Cross-validation** | 🟡 Medium | Replace single train/test split with k-fold CV for more reliable metric estimates |
| **Treat as classification** | 🟡 Medium | Bin quality into Low/Medium/High and use a classifier — may be more useful for production decisions |
| **SHAP explanations** | 🟢 Low | Add feature importance via SHAP to explain predictions to end users |

### 🏗️ MLOps & Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Experiment tracking** | Integrate MLflow to log params, metrics, and model versions across runs |
| **Data versioning** | Add DVC to version `Drinks-data.csv` and pipeline outputs |
| **Fix random state in split** | `train_test_split(data)` has no `random_state` — results vary each run; add `random_state=42` |
| **Pipeline validation gate** | Check `status.txt` before proceeding to transformation; abort if `False` |
| **Unit tests** | Add `pytest` tests for each component; test schema validation, split ratios, metric computation |
| **Async retraining** | Replace `os.system("python main.py")` in `/train` with a background task (Celery or threading) |
| **API documentation** | Add Swagger/OpenAPI docs via Flask-RESTX |

### 📦 Product Improvements

- Add a **confidence indicator** showing the model's uncertainty around the prediction
- Display **feature contribution breakdown** — which inputs most influenced the quality score
- Add **sample data buttons** for common wine profiles to help users understand typical input ranges
- Support **batch prediction** — upload a CSV of multiple samples, download predictions
- Add **model drift monitoring** — alert when incoming data distribution shifts from training data

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `ModuleNotFoundError: mlProject` | Run `pip install -r requirements.txt` which installs package via `-e .` |
| `FileNotFoundError: model.joblib` | Run `python main.py` to execute the full pipeline and generate the model |
| `FileNotFoundError: config/config.yaml` | Run all scripts from the project root directory, not from inside `src/` |
| `Validation status: False` | Column names in CSV don't match `schema.yaml` — check downloaded data source |
| Port 8080 already in use | `lsof -ti:8080 \| xargs kill -9` or change port in `app.py` |
| Docker build fails | Ensure Docker Desktop is running; check internet connectivity for `pip install` |
| EC2 cannot pull from ECR | Attach `AmazonEC2ContainerRegistryFullAccess` policy to EC2 IAM role |
| `/train` route hangs | `os.system("python main.py")` runs synchronously — allow 1–2 minutes for full pipeline |
| Prediction returns unexpected values | Ensure input values are in the correct numeric range (see feature table in Section 6) |
| GitHub Actions deploy fails | Check EC2 self-hosted runner is online: GitHub → Settings → Actions → Runners |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **ElasticNet** | Linear regression model combining L1 (Lasso) and L2 (Ridge) regularisation, controlled by `alpha` and `l1_ratio` |
| **L1 regularisation (Lasso)** | Adds absolute value of coefficients to loss function — can drive some coefficients to zero (feature selection) |
| **L2 regularisation (Ridge)** | Adds squared value of coefficients to loss function — shrinks all coefficients uniformly |
| **alpha** | Controls overall regularisation strength in ElasticNet — higher values = more regularisation |
| **l1_ratio** | Proportion of L1 vs L2 in ElasticNet: 0 = pure Ridge, 1 = pure Lasso, 0.1 = 10% L1 + 90% L2 |
| **RMSE** | Root Mean Squared Error — penalises large prediction errors more than MAE |
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual values |
| **R²** | Coefficient of determination — proportion of variance in target explained by the model (1.0 = perfect) |
| **ConfigurationManager** | Central class that reads all 3 YAML files and returns typed config dataclasses per stage |
| **ConfigBox** | `python-box` class that allows dot-notation access to YAML/dict content (e.g. `config.data_ingestion.root_dir`) |
| **frozen dataclass** | Python dataclass with `frozen=True` — all fields are immutable after creation; used for all config entities |
| **`ensure` annotations** | Python library that validates function argument types at runtime using `@ensure_annotations` decorator |
| **joblib** | Serialisation library optimised for NumPy arrays and sklearn models — faster than pickle for ML objects |
| **PredictionPipeline** | Lightweight class that loads `model.joblib` at init and exposes a `predict(data)` method |
| **Physicochemical** | Properties measured by physical or chemical analysis (e.g. pH, acidity, density) as opposed to sensory tasting |
| **Volatile acidity** | Acetic acid content in wine — high levels produce an unpleasant vinegar-like taste |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---


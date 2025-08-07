
# 🚀 Car Price Prediction - End-to-End ML Project (MLOps + Deployment)

Welcome to the **Car Price Prediction** project – a complete **MLOps-based ML pipeline**, built using industry-standard tools and deployed at scale using **AWS services**, **Docker**, and **GitHub Actions**.  
This repository demonstrates how to build, train, track, and deploy a machine learning model with full CI/CD and cloud integration.

---

## 🛠️ Tech Stack Used

| Category              | Tools & Technologies                                      |
|----------------------|-----------------------------------------------------------|
| **Database**          | `MongoDB`                                                 |
| **ML Tracking**       | `MLflow`, `Dagshub`                                       |
| **Data Versioning**   | `DVC`, `S3` (AWS)                                         |
| **Model Training**    | `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` |
| **Model Deployment**  | `Flask`, `Docker`, `ECR`, `AWS`                           |
| **CI/CD Pipeline**    | `GitHub Actions`                                          |
| **Cloud Infrastructure** | `AWS CLI`, `IAM`, `S3`, `ECR`, `kubectl`, `eksctl`     |

---

## 📁 Project Structure

```
📦 car-price-prediction/
├── .github/workflows/         # CI/CD configuration
├── dvc.yaml                   # DVC pipeline configuration
├── params.yaml                # Hyperparameters for the pipeline
├── local_s3/                  # Local cache directory for DVC
├── flask_app/                 # Flask-based ML model API
│   ├── app.py                 # Main Flask application
│   ├── templates/             # HTML templates
│   ├── static/                # CSS/JS files (if any)
├── src/                       # Core pipeline code
│   ├── logger.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── data_transforming.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── model_register.py
├── tests/                     # Test scripts
├── scripts/                   # Utility/automation scripts
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Problem Statement

Build a machine learning model to predict the **selling price of a car** based on various features like:
- Year, Fuel Type, Transmission, Kilometers Driven, Owner Type, etc.

---

## 🧪 ML Workflow Overview

1. **Data Ingestion** - Load and clean raw CSV files.
2. **Data Preprocessing** - Handle missing values, feature transformation.
3. **Data Transforming** - Generate new features or encode categorical data.
4. **Model Training** - Train ML model (RandomForest, XGBoost, etc.)
5. **Model Evaluation** - Evaluate using RMSE, MAE, R².
6. **Model Registration** - Register best model with MLflow.

---

## 🔁 Reproducible Pipeline using DVC

We’ve used **DVC** to version control data, model, and pipeline stages.

```bash
dvc init
dvc repro             # Run entire pipeline
dvc status            # See changes in pipeline/data
dvc push              # Push artifacts to remote (S3/local)
```

---

## 🧠 MLflow Integration (via Dagshub)

- Each model training run is logged via **MLflow Tracking UI** hosted on **Dagshub**.
- Easily visualize parameters, metrics, and artifacts.
- Enables experiment tracking and model comparison.

🔗 [MLflow UI on Dagshub](https://dagshub.com/your-username/your-repo-name.mlflow)

---

## 🐳 Dockerized Flask App

- Lightweight Flask app containerized using Docker
- Predict prices via browser form UI

### ⚙️ Build and Run Locally

```bash
docker build -t capstone-app:latest .
docker run -p 8888:5000 -e CAPSTONE_TEST=your_dagshub_token capstone-app:latest
```

---

## 🔄 CI/CD using GitHub Actions

**Workflow Highlights:**

- On each push:
  - Code is linted and tested.
  - DVC stages are run if any change is detected.
  - Model is trained and logged to MLflow.
  - Docker image is built and pushed to AWS ECR.

✅ Auto model training  
✅ Auto deployment to cloud  
✅ Auto versioning and logging

---

## ☁️ AWS Integration

### 🔐 Secrets Stored in GitHub

| Secret Name              | Description                      |
|--------------------------|----------------------------------|
| AWS_ACCESS_KEY_ID        | IAM user's access key            |
| AWS_SECRET_ACCESS_KEY    | IAM user's secret key            |
| AWS_REGION               | Deployment region (e.g., ap-south-1) |
| ECR_REPOSITORY           | ECR repository name              |
| AWS_ACCOUNT_ID           | AWS account ID                   |
| CAPSTONE_TEST            | Dagshub MLflow auth token        |

### 💾 S3 + ECR
- **S3**: Used as remote cache for DVC
- **ECR**: Used to store docker images

---

## 📸 Output Screenshots

- 📍 MLflow Tracking on Dagshub
- 📍 Flask App UI
- 📍 GitHub Actions CI Pipeline

---

## ✅ How to Run Locally

```bash
# Clone repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create env and install dependencies
conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt

# Run DVC pipeline
dvc repro

# Run Flask App
cd flask_app
python app.py
```

---

## 📦 Future Enhancements

Use dataset which contains images of cars. Use clustering or similarity models to suggest cars under budget and available nearby.

---

## 🙌 Author

👤 **Sourav Bhardwaj**  
📬 [Contact on LinkedIn](https://www.linkedin.com/in/sourav-bhardwaj-88b9b7212/)  
📧 Email: bhardwajsourav113@gmail.com

---

## ⭐ Final Note

This project is a complete demonstration of **production-grade machine learning engineering** using cutting-edge MLOps tools.  
**Interviewers** can explore:
- Real-world CI/CD practices
- Automated deployment pipelines
- MLflow logging & DVC versioning
- Docker & AWS Services

> ✅ If you're looking for a candidate who understands **both ML and deployment lifecycle**, you're in the right place.

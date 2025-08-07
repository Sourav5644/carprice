# 🚗 Car Price Prediction App

A complete end-to-end Machine Learning project that predicts the price of a used car based on multiple features. This project is containerized using Docker, tracks data/model using DVC, and exposes a REST API using FLASK for real-time predictions.

---

## 🌟 Features

- Predict car resale price based on features like year, km driven, fuel type, etc.
- Scalable and reproducible ML workflow using DVC
- Flask-powered backend for real-time inference
- Containerized with Docker for easy deployment
- Cloud-ready configuration (e.g., Render.com deployment support)
- Modular and production-grade project structure

---

## 🛠️ Tech Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Web Framework | FastAPI |
| ML Libraries | scikit-learn, pandas, numpy |
| Version Control | Git |
| Data & Pipeline Versioning | DVC |
| Containerization | Docker |
| Deployment | Render (via `render.yaml`) |
| Configuration | YAML (`params.yaml`) |
| Package Management | `pip`, `requirements.txt`, `setup.py` |

---

## 📁 Project Structure

```
carproj/
└── carprice/
    ├── app.py               # FastAPI app entry point
    ├── demo.py              # Demo code for testing
    ├── template.py          # Prediction and preprocessing logic
    ├── params.yaml          # ML parameters
    ├── requirements.txt     # Python dependencies
    ├── Dockerfile           # Docker configuration
    ├── dvc.yaml             # DVC pipeline definition
    ├── dvc.lock             # DVC lock file
    ├── .dvc/                # DVC metadata and cache
    ├── render.yaml          # Deployment config (e.g., Render.com)
    ├── setup.py             # Project packaging
    └── cred.txt             # (Avoid committing secrets!)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd carproj/carprice
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run FastAPI app locally

```bash
uvicorn app:app --reload
```

Open browser at: `http://127.0.0.1:8000/docs` to test the API.

---

## 🐳 Run with Docker

```bash
docker build -t carprice-app .
docker run -p 8888:5000 carprice-app
```

Then go to: `http://localhost:8888/docs`

---

## 🔁 ML Pipeline using DVC

This project uses DVC to version control data and model artifacts.

- `dvc.yaml` defines the ML pipeline stages (data prep → train → evaluate)
- `params.yaml` holds the hyperparameters
- `.dvc/cache/` manages versioned data and models

### Basic DVC commands

```bash
dvc repro            # Re-run pipeline
dvc metrics show     # Show model performance
dvc plots diff       # Compare metrics across versions
```

---

## 📊 Model Overview

The model predicts car prices based on:

- Car make year
- Fuel type
- Ownership history
- Transmission type
- Kilometers driven

Preprocessing includes categorical encoding and scaling. The model is trained using scikit-learn regressors (can be extended with hyperparameter tuning and RandomForestRegressor, etc.)

---

## ☁️ Deployment

- Deployment-ready with `Dockerfile` and `render.yaml` for Render.com or other platforms.
- To deploy, connect GitHub repo with Render and use `render.yaml` config.

---

## ✅ Future Improvements

- Add frontend UI (e.g., React or simple HTML form)
- Extend to support other prediction targets (e.g., mileage prediction)
- Integrate MLflow or Weights & Biases for experiment tracking
- Use MongoDB or PostgreSQL to store predictions

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

Thanks to open-source communities and FastAPI, scikit-learn, DVC, and Docker contributors.

---

## 👨‍💻 Author

**Sourav Bhardwaj**  
Passionate ML Developer | AWS, Docker, FastAPI Enthusiast  
[LinkedIn](https://www.linkedin.com/in/sourav-bhardwaj-88b9b7212/) • [GitHub](https://github.com/Sourav5644)

---

## 📬 Contact
Gmial:bhardwajsourav113@gmial.com

For queries or feedback, feel free to reach out!


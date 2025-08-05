from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import time

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import dagshub
import warnings
warnings.filterwarnings("ignore")


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Sourav5644"
repo_name = "carprice"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------



# ----------------------------------------
# MLflow tracking setup with DagsHub
# ----------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Sourav5644/carprice.mlflow')
# dagshub.init(repo_owner='Sourav5644', repo_name='carprice', mlflow=True)

# ----------------------------------------
# Flask app setup
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# Prometheus metrics
# ----------------------------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ----------------------------------------
# Helper function: Get latest model version
# ----------------------------------------
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        all_versions = client.search_model_versions(f"name = '{model_name}'")
        if not all_versions:
            raise ValueError(f"No versions of model '{model_name}' registered.")
        # fallback to max version
        version = max(int(v.version) for v in all_versions)
        return str(version)
    return versions[0].version

# ----------------------------------------
# Load Model and Preprocessing Pipeline
# ----------------------------------------
model_name = "my_model"
model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

pipeline = pickle.load(open('data/processed/preprocessor.pkl', 'rb'))

# ----------------------------------------
# Routes
# ----------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    # -----------------------
    # Read form data
    # -----------------------
    form_data = request.form

    accessories = request.form.getlist('Accessories')
    accessories_str = ",".join(accessories)

    data = {
        'Company Name': [form_data.get("Company")],
        'Car Name': [form_data.get("CarName")],
        'Fuel Type': [form_data.get("Fuel Type")],
        'Tyre Condition': [form_data.get("Tyre Condition")],
        'Make Year': [int(form_data.get("Make Year"))],
        'Owner Type': [form_data.get("Owner Type")],
        'Total_KM_Run': [int(form_data.get("Total_KM_Run"))],
        'Transmission Type': [form_data.get("Transmission Type")],
        'Service Record': [int(form_data.get("Service Record"))],
        'Insurance': [int(form_data.get("Insurance"))],
        'Registration Certificate': [int(form_data.get("Registration Certificate"))],
        'Accessories': [accessories_str],
        'State Name': [form_data.get("State Name")]
    }

    input_df = pd.DataFrame(data)

    # -----------------------
    # Apply Preprocessing
    # -----------------------
    features = pipeline.transform(input_df)

    # -----------------------
    # Predict
    # -----------------------
    prediction = model.predict(features)[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    # -----------------------
    # Render result
    # -----------------------
    formatted_prediction = f"Predicted Price: ₹{int(prediction):,}".replace(",", ",")
    return render_template("index.html", result=formatted_prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# ----------------------------------------
# Run the App
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

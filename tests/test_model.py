import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import r2_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "vikashdas770"
        repo_name = "YT-Capstone-Project"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load model from MLflow
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the preprocessor
        cls.preprocessor = pickle.load(open('data/processed/preprocessor.pkl', 'rb'))

        # Load test data
        cls.holdout_data = pd.read_csv('data/processed/test_processed.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        sample_input = {
            'Car Name': 'Swift',
            'Fuel Type': 'Petrol',
            'Make Year': 2020,
            'Owner Type': 'First',
            'Total KM Run': 30000,
            'Transmission Type': 'Manual',
            'Service Record': 'Yes',
            'Insurance': 'Comprehensive',
            'Registration Certificate': 'Original',
            'Accessories': 'Yes',
            'State Name': 'Delhi',
            'Tyre Condition': 'New'
        }
        input_df = pd.DataFrame([sample_input])
        input_data = self.preprocessor.transform(input_df)

        prediction = self.new_model.predict(input_data)

        self.assertEqual(input_data.shape[1], len(self.preprocessor.get_feature_names_out()))
        self.assertEqual(len(prediction), input_data.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # For regression output

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Apply preprocessing
        X_transformed = self.transform(X_holdout)

        # Predict
        y_pred = self.new_model.predict(X_transformed)

        # Evaluate
        r2 = r2_score(y_holdout, y_pred)

        expected_r2_score = 0.40
        self.assertGreaterEqual(r2, expected_r2_score, f'R2 score should be at least {expected_r2_score}')

if __name__ == "__main__":
    unittest.main()

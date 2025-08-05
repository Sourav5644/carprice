import numpy as np
import pandas as pd
import os
import sys
import yaml
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.logger import logging

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Failed to load data: %s', e)
        raise

def save_data(array: np.ndarray, path: str):
    """Save the transformed data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(array).to_csv(path, index=False)
    logging.info(f"Saved processed data to {path}")

def make_pipeline(df: pd.DataFrame) -> Pipeline:
    try:
        logging.info("Creating preprocessing pipeline...")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        logging.info(f"Numerical columns: {num_cols}")
        logging.info(f"Categorical columns: {cat_cols}")

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, num_cols),
            ('cat', categorical_pipeline, cat_cols)
        ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        logging.info("Pipeline created successfully.")
        return pipeline

    except Exception as e:
        logging.error('Error in creating pipeline: %s', e)
        raise

def main():
    try:
        # Load params
        params = load_params('params.yaml')
        target = params['target']['column'] 


        # Load data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Split features and target
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]

        X_test = test_data.drop(columns=[target])
        y_test = test_data[target]

        logging.info("Train and test datasets loaded and split.")

        # Create and fit pipeline
        preprocessor = make_pipeline(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Combine with target
        train_final_array = np.c_[X_train_transformed, y_train.to_numpy()]
        test_final_array = np.c_[X_test_transformed, y_test.to_numpy()]

        # Save processed data
        save_data(train_final_array, './data/processed/train_processed.csv')
        save_data(test_final_array, './data/processed/test_processed.csv')

        # Save pipeline
        with open('./data/processed/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info("Preprocessing pipeline saved to preprocessor.pkl")

    except Exception as e:
        logging.error("Failed during feature engineering: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
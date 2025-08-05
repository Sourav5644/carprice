import os
import numpy as np
import pandas as pd
from src.logger import logging



def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info(f'Original data shape: {df.shape}')

        # Fix Eeco car price
        df.loc[df['Car Name'].str.contains('Eeco', case=False, na=False), 'Price'] /= 10
        logging.info("Adjusted price for Eeco cars.")

        # Extract and map state code to state name
        df['State Code'] = df['Registration Number'].str.slice(0, 2)
        state_map = {
            'KA': 'Karnataka', 'AP': 'Andhra Pradesh', 'TS': 'Telangana', 'DL': 'Delhi',
            'MH': 'Maharashtra', 'KL': 'Kerala', 'TN': 'Tamil Nadu', 'PB': 'Punjab',
            'MP': 'Madhya Pradesh', 'JS': 'Jharkhand'
        }
        df['State Name'] = df['State Code'].map(state_map)
        logging.info('succesfull fetch State Name')

        # Map Insurance values
        insurance_map = {
            "No Current Insurance": "No Insurance",
            "Valid Until [date]": "Insurance Valid"
        }
        df['Insurance'] = df['Insurance'].map(insurance_map).fillna("No Insurance")
        logging.info('Insurance value updated')

        # Process Service Record using np.select
        conditions = [
            df['Service Record'].isna() | (df['Service Record'] == 'No Service Record'),
            (df['Service Record'] == 'Full Service History') | df['Service Record'].str.startswith('Major Service', na=False)
        ]
        choices = ['Not Available', 'Available']
        df['Service Record'] = np.select(conditions, choices, default='Not Available')
        logging.info('updated servive record successfully')

        # Map Registration Certificate
        registration_map = {
            "Not Available": "No",
            "Available": "Yes"
        }
        df['Registration Certificate'] = df['Registration Certificate'].map(registration_map).fillna("No")
        logging.info('updated registration certiicate successfully')

        # Rename columns
        df.rename(columns={'Mileage': 'Total_KM_Run'}, inplace=True)
        logging.info('rename mileage successfully')

        # Drop unnecessary columns
        columns_to_drop = ['Registration Number', 'Body Color', 'Unnamed: 0', 'Engine Number', 'Chassis Number','State Code']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        logging.info('successfully drop the colomns')

        logging.info(f"Processed data shape: {df.shape}")
        logging.info(f'After pre processing column_names:{df.columns}')
        return df

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def main():
    try:
        logging.info('Fetching the data from data/raw')
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('Train and test datasets loaded successfully')

        logging.info('For training dataset')
        train_processed = preprocess_dataframe(train_data)
        logging.info('-----------------------------------------------------------------------')
        logging.info('=======================================================================')
        logging.info('For testing data')
        test_processed = preprocess_dataframe(test_data)


        data_path = os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logging.info('Processed data saved to %s', data_path)

    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()

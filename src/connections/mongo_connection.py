
from pymongo import MongoClient
import pandas as pd
import logging

def get_mongo_data(uri: str, database: str, collection: str) -> pd.DataFrame:
    try:
       
        client = MongoClient(uri)
        db = client[database]
        collection = db[collection]
        data = list(collection.find())
        print(f"Number of documents fetched: {len(data)}")
        df = pd.DataFrame(data)

        # Drop MongoDB's default _id field if present
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)

        logging.info("Data fetched successfully from MongoDB")
        return df
    except Exception as e:
        logging.error("Error fetching data from MongoDB: %s", e)
        raise

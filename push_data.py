import os 
import sys
import json
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv

from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

# Load environment variables
load_dotenv()
uri = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def cv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(inplace=True, drop=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def insert_data_mongo(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(uri, tlsCAFile=ca)
            db = self.mongo_client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == "__main__":
    FILE_PATH = r"Network_Data/phisingData.csv"
    DATABASE_NAME = "NetworkSecurity"
    COLLECTION_NAME = "NetworkData"
    
    try:
        networkobj = NetworkDataExtract()
        records = networkobj.cv_to_json_converter(file_path=FILE_PATH)
        logger.info(f"Total records converted from CSV: {len(records)}")
        
        no_of_records = networkobj.insert_data_mongo(records, DATABASE_NAME, COLLECTION_NAME)
        logger.info(f"Successfully inserted {no_of_records} records into MongoDB.")
        print(f"Inserted {no_of_records} records into MongoDB.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")

import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "CLASS_LABEL"
PIPELINE_NAME: str = "network_security"
ARTIFACT_DIR: str = "artifact"
FILENAME: str = "phishingData.csv"

TRAIN_FILENAME: str = "train.csv"
TEST_FILENAME: str = "test.csv"

"""
DATA Ingestion related constant starts with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "NetworkSecurity"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
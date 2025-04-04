from NetworkSecurityFun.components.data_ingestion import DataIngestion
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.entity.config_entity import DataIngestionConfig
from NetworkSecurityFun.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        dataingestion=DataIngestion(dataingestionconfig)
        logger.info("Data Ingestion started")
        dataingestionartifact=dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
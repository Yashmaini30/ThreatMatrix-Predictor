from NetworkSecurityFun.components.data_ingestion import DataIngestion
from NetworkSecurityFun.components.data_validation import DataValidation
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.entity.config_entity import DataIngestionConfig,DataValidationConfig
from NetworkSecurityFun.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(data_ingestion_config)
        logger.info("Data Ingestion started")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logger.info("Data Ingestion completed")
        
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logger.info("Data Validation started")
        data_validation_artifact=data_validation.initiate_data_validation()
        logger.info("Data Validation completed")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
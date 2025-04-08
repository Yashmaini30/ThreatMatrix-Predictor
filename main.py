from NetworkSecurityFun.components.data_ingestion import DataIngestion
from NetworkSecurityFun.components.data_validation import DataValidation
from NetworkSecurityFun.components.data_transformation import DataTransformation
from NetworkSecurityFun.components.model_trainer import ModelTrainer
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
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

        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logger.info("Data Transformation started")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logger.info("Data Transformation completed")

        logger.info("Model Training started")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config,data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logger.info("Model Training completed")
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
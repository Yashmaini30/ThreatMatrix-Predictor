import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurityFun.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurityFun.constants.training_pipeline import TARGET_COLUMN

from NetworkSecurityFun.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from NetworkSecurityFun.entity.config_entity import DataTransformationConfig
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_transformer_object(cls) -> Pipeline:
        """
        It initilizes a KNNimputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the imputer object

        Args:
            cls: DataTransformation class

        Returns:
            Pipeline
        """
        logger.info(f"Entered the get_transformer_object method of DataTransformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(f"Imputer object created. Exited the get_transformer_object method of DataTransformation class")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info(f"Entered the fit_transform method of DataTransformation class")
        try:
            logger.info(f"Reading train file: [{self.data_validation_artifact.valid_train_file_path}]")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## transform the data
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            ## transform test data
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessor=self.get_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_features = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]

            ## save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            ## save preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_models/preprocessor.pkl", preprocessor_object)

            ## preparing artifact
            data_transforrmation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transforrmation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)     
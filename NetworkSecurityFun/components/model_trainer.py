import  os
import sys

from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

from NetworkSecurityFun.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from NetworkSecurityFun.entity.config_entity import ModelTrainerConfig

from NetworkSecurityFun.utils.ml_utils.model.estimator import NetworkSecurityModel
from NetworkSecurityFun.utils.main_utils.utils import save_object,load_object
from NetworkSecurityFun.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from NetworkSecurityFun.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import mlflow
import dagshub
from dotenv import load_dotenv
load_dotenv()

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO"),
    mlflow=True,
    auth_token=os.getenv("DAGSHUB_PASSWORD")
)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def track_model(self,best_model,best_model_name,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)

            mlflow.log_param("model_name", best_model_name)
            mlflow.log_params(best_model.get_params())
            mlflow.sklearn.log_model(best_model,"Model")
    
    def train_model(self,x_train,y_train,x_test,y_test):
        models = {
            "Logistic Regression": LogisticRegression(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Support Vector Machine": SVC(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        }
        params={
            "Logistic Regression": {
                "C": [0.1, 1, 10, 100, 1000],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "newton-cg", "lbfgs"],
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy","log_loss"],
                "splitter": ["best", "random"],
                "max_features": [None, "sqrt", "log2"],
            },
            "Support Vector Machine": {
                "C": [0.1, 1, 10, 100, 1000],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "gamma": ["scale", "auto"],
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            },
            "Random Forest": {
                "n_estimators": [10, 50, 100, 200],
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "max_depth": [3, 5, 7],
            }
        }
        model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                         models=models,param=params)
        
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        best_params = best_model.get_params()

        if best_model_score < self.model_trainer_config.expected_accuracy:
            raise Exception("No best model found")

        y_train_pred = best_model.predict(x_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric= get_classification_score(y_true=y_test,y_pred=y_test_pred)

        ### track the mlflow
        self.track_model(best_model=best_model,best_model_name=best_model_name,classificationmetric=classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

        Network_Model=NetworkSecurityModel(
            preprocessor=preprocessor,
            model=best_model
        )
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        save_object("final_models/model.pkl",best_model)
        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            trained_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
            best_model_name=best_model_name,
            best_model_params=best_params)
        
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        logger.info(f"Best model selected: {best_model_name} with parameters: {best_params}")


        return model_trainer_artifact
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            ## loading train and test array
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
import os
import sys
import warnings
from typing import Any, Dict, Tuple

from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

from NetworkSecurityFun.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from NetworkSecurityFun.entity.config_entity import ModelTrainerConfig

from NetworkSecurityFun.utils.ml_utils.model.estimator import NetworkSecurityModel
from NetworkSecurityFun.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models, 
)
from NetworkSecurityFun.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import f1_score

import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.auth.add_app_token(os.getenv("DAGSHUB_AUTH_TOKEN"))
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO"),
    mlflow=True,
)


class ModelTrainer:
    """Train a collection of models, pick the best F‑score, track with MLflow."""

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    @staticmethod
    def _track_model(name: str, model: Any, metric_obj: Any) -> None:
        with mlflow.start_run():
            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score", metric_obj.f1_score)
            mlflow.log_metric("precision", metric_obj.precision_score)
            mlflow.log_metric("recall", metric_obj.recall_score)
            mlflow.sklearn.log_model(model, "model")

    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> ModelTrainerArtifact:

        models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=10_000, solver="saga", n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(),
            "Support Vector Machine": SVC(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

        params: Dict[str, Any] = {
            "Logistic Regression": [
                {
                    "penalty": ["elasticnet"],
                    "l1_ratio": [0.5, 0.8],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["saga"],
                },
                {
                    "penalty": ["l1", "l2"],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "saga"],
                },
            ],
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 10, 20, 30],
            },
            "Support Vector Machine": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
            "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
            },
            "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 1]},
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.05],
                "max_depth": [3, 5],
            },
        }
        scores: Dict[str, float] = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
        best_estimator = models[best_name]  

        expected = self.model_trainer_config.expected_accuracy
        if best_score < expected:
            warnings.warn(
                f"Best F1 {best_score:.3f} below expected {expected:.3f}",
                RuntimeWarning,
            )
        train_pred = best_estimator.predict(X_train)
        test_pred = best_estimator.predict(X_test)
        train_metric = get_classification_score(y_train, train_pred)
        test_metric = get_classification_score(y_test, test_pred)

        self._track_model(best_name, best_estimator, test_metric)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        full_pipeline = NetworkSecurityModel(preprocessor=preprocessor, model=best_estimator)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, full_pipeline)
        save_object("final_models/model.pkl", best_estimator)

        artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            trained_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
            best_model_name=best_name,
            best_model_params=best_estimator.get_params(),
        )
        logger.info(f"🏆  Best model: {best_name} – F1={best_score:.4f}")
        return artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

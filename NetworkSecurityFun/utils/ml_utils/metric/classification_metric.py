from NetworkSecurityFun.entity.artifact_entity import ClassifierMetricArtifact
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, recall_score

def get_classification_score(y_true, y_pred) -> ClassifierMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        
        classifier_metric = ClassifierMetricArtifact(f1_score=model_f1_score,
                                                    precision_score=model_precision_score,
                                                    recall_score=model_recall_score)
        return classifier_metric
    except Exception as e:
        raise NetworkSecurityException(e, sys)
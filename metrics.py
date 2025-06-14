import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(y_true, y_pred):
    """Return accuracy, precision, recall and f1 as a dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def metrics_to_dataframe(metrics_dict):
    """Convert metrics dictionary to pandas DataFrame for plotting."""
    return pd.DataFrame([metrics_dict])
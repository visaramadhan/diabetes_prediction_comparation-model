import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
    }
    
    # Tambahkan ROC AUC jika probabilitas tersedia
    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    
    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics
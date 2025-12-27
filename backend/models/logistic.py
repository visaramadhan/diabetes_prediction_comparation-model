import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.name = "Logistic Regression"
    
    def evaluate(self, X, y):
        # Definisikan scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        # Lakukan cross-validation
        cv_results = cross_validate(self.model, X, y, cv=5, scoring=scoring, return_train_score=False)
        
        # Hitung rata-rata metrik
        metrics = {
            'accuracy': float(np.mean(cv_results['test_accuracy'])),
            'precision': float(np.mean(cv_results['test_precision'])),
            'recall': float(np.mean(cv_results['test_recall'])),
            'f1': float(np.mean(cv_results['test_f1'])),
            'roc_auc': float(np.mean(cv_results['test_roc_auc']))
        }
        
        return metrics
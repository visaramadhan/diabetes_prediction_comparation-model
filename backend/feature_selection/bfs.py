import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


class BorutaSelector:
    def __init__(self):
        self.estimator = RandomForestClassifier(random_state=42)
        self.selector = BorutaPy(self.estimator, n_estimators='auto', verbose=0, random_state=42)
        self.feature_names = None
    
    def select_features(self, X, y):
        self.feature_names = X.columns
        # Boruta membutuhkan array numpy
        X_np = X.values
        
        # Fit Boruta
        self.selector.fit(X_np, y)
        
        # Pilih fitur yang terpilih (confirmed dan tentative)
        selected_features = X.columns[self.selector.support_]
        X_selected = X[selected_features]
        
        return X_selected
    
    def get_selected_features(self):
        if self.selector is None or self.feature_names is None:
            return []
        
        return list(self.feature_names[self.selector.support_])
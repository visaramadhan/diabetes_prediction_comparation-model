import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


class RFESelector:
    def __init__(self):
        self.estimator = RandomForestClassifier(random_state=42)
        self.selector = None
        self.feature_names = None
    
    def select_features(self, X, y, n_features_to_select=None):
        # Jika n_features_to_select tidak ditentukan, gunakan setengah dari jumlah fitur
        if n_features_to_select is None:
            n_features_to_select = X.shape[1] // 2
        
        self.feature_names = X.columns
        self.selector = RFE(self.estimator, n_features_to_select=n_features_to_select)
        self.selector.fit(X, y)
        
        # Pilih fitur yang terpilih
        selected_features = X.columns[self.selector.support_]
        X_selected = X[selected_features]
        
        return X_selected
    
    def get_selected_features(self):
        if self.selector is None or self.feature_names is None:
            return []
        
        return list(self.feature_names[self.selector.support_])
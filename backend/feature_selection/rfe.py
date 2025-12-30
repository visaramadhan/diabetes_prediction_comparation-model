import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


class RFESelector:
    def __init__(self):
        self.estimator = RandomForestClassifier(random_state=42)
        self.selector = None
        self.feature_names = None
        self._ranking_ = None
        self._importance_map = None
    
    def select_features(self, X, y, n_features_to_select=None):
        if n_features_to_select is None:
            n_features_to_select = X.shape[1] // 2
        
        self.feature_names = X.columns
        self.selector = RFE(self.estimator, n_features_to_select=n_features_to_select)
        self.selector.fit(X, y)
        self._ranking_ = self.selector.ranking_
        
        selected_features = X.columns[self.selector.support_]
        X_selected = X[selected_features]
        
        try:
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_selected, y)
            importances = rf.feature_importances_
            self._importance_map = {f: float(importances[i]) for i, f in enumerate(selected_features)}
        except Exception:
            self._importance_map = {f: 0.0 for f in selected_features}
        
        return X_selected
    
    def get_selected_features(self):
        if self.selector is None or self.feature_names is None:
            return []
        
        return list(self.feature_names[self.selector.support_])
    
    def get_feature_scores(self):
        if self.selector is None or self.feature_names is None:
            return []
        scores = []
        for i, f in enumerate(self.feature_names):
            rank = int(self._ranking_[i]) if self._ranking_ is not None else None
            imp = float(self._importance_map.get(f, 0.0)) if self._importance_map is not None else 0.0
            scores.append({'feature': str(f), 'ranking': rank, 'importance': imp})
        scores.sort(key=lambda x: (x['ranking'] if x['ranking'] is not None else 9999, -x['importance']))
        return scores

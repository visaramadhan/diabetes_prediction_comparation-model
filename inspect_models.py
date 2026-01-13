import joblib
import pandas as pd
import os

model_dir = r"c:\Users\Visa Ramadhan\Documents\web\prediction_dm\model"

def inspect_pickle(filename):
    path = os.path.join(model_dir, filename)
    print(f"\n--- Inspecting {filename} ---")
    try:
        obj = joblib.load(path)
        
        print(f"Type: {type(obj)}")
        
        if hasattr(obj, 'feature_names_in_'):
            print("Feature names in:", obj.feature_names_in_)
        elif hasattr(obj, 'transformers_'):
            print("Transformers:", obj.transformers_)
            # For ColumnTransformer
            for name, trans, cols in obj.transformers_:
                print(f"  Transformer {name}: {cols}")
        elif hasattr(obj, 'steps'):
            print("Pipeline steps:", [s[0] for s in obj.steps])
            
        if hasattr(obj, 'n_features_in_'):
            print("N features in:", obj.n_features_in_)
            
    except Exception as e:
        print(f"Error loading {filename}: {e}")

inspect_pickle("preprocessor.pkl")
inspect_pickle("rf_model_baseline.pkl")
inspect_pickle("lr_model_rfe.pkl")
inspect_pickle("rfe_selector.pkl")

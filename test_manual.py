import json
import joblib
import pandas as pd
import os
import sys

# Add backend to path to import app logic if needed, but easier to test via requests if server running.
# Or just import the logic directly.

# Let's try to import the loading logic from app.py to see if it crashes.
sys.path.append(os.path.join(os.getcwd(), 'backend'))

print("Attempting to import app...")
try:
    from app import load_manual_models, manual_models, predict_manual
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    # Might fail due to flask imports if environment issues, but let's try.
    # If fails, I will trust the code edits.

print("Testing model loading...")
# load_manual_models() is called on import of app.py usually, but let's check manual_models dict
print(f"Loaded models: {list(manual_models.keys())}")

if 'rf_baseline' in manual_models:
    print("RF Baseline loaded.")
    model = manual_models['rf_baseline']
    print(f"Model type: {type(model)}")
else:
    print("RF Baseline NOT loaded.")

if 'lr_rfe' in manual_models:
    print("LR RFE loaded.")
else:
    print("LR RFE NOT loaded.")

# Test prediction logic (simulation)
print("\nTesting Prediction Logic...")
# Mock request
class MockRequest:
    def __init__(self, json_data):
        self.json = json_data

# We need to mock flask.request
import flask
# This is hard to mock without context.

# Instead, let's just use the model directly to predict on dummy data
if 'rf_baseline' in manual_models:
    model = manual_models['rf_baseline']
    # RF expects 18 cols
    cols = ['Usia', 'Jenis_Kelamin', 'Kehamilan', 'Poliuria', 'Polidipsia', 'penurunan_bb', 
               'mudah_lelah', 'Polifagia', 'infeksi', 'penglihatan_kabur', 'Gatal', 'Irritability', 
               'penyembuhan_lambat', 'kesemutan', 'kekakuan otot', 'kerontokan_rambut', 'genetik_diabetes', 'obesitas']
    
    row = {c: 0 for c in cols}
    row['Usia'] = 45
    row['Jenis_Kelamin'] = 1 # Laki-laki
    row['Poliuria'] = 1 # Yes
    
    X = pd.DataFrame([row])
    X = X[cols] # Ensure order
    
    try:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        print(f"RF Prediction: {pred}, Prob: {prob}")
    except Exception as e:
        print(f"RF Prediction failed: {e}")

if 'lr_rfe' in manual_models:
    model = manual_models['lr_rfe']
    # LR expects: ['Usia', 'Jenis_Kelamin_P', 'Kehamilan_Yes', 'Polidipsia_Yes', 'penurunan_bb_Yes', 'mudah_lelah_Yes', 'penglihatan_kabur_Yes', 'kesemutan_Yes']
    cols = ['Usia', 'Jenis_Kelamin_P', 'Kehamilan_Yes', 'Polidipsia_Yes', 'penurunan_bb_Yes', 'mudah_lelah_Yes', 'penglihatan_kabur_Yes', 'kesemutan_Yes']
    
    row = {c: 0 for c in cols}
    row['Usia'] = 50
    row['Jenis_Kelamin_P'] = 0 # Laki-laki
    row['Polidipsia_Yes'] = 1
    
    X = pd.DataFrame([row])
    X = X[cols]
    
    try:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        print(f"LR Prediction: {pred}, Prob: {prob}")
    except Exception as e:
        print(f"LR Prediction failed: {e}")

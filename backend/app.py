from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Import model dan fungsi seleksi fitur
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.logistic import LogisticRegressionModel
from feature_selection.rfe import RFESelector
from feature_selection.bfs import BorutaSelector
from utils.metrics import calculate_metrics

app = Flask(__name__)
CORS(app)

# Konfigurasi upload file
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'models')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Pastikan direktori upload ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Sesi training di memori
sessions = {}

# Koneksi PostgreSQL (opsional)
def get_db_connection():
    dsn = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/prediction_dm')
    try:
        conn = psycopg2.connect(dsn)
        return conn
    except Exception:
        return None

def init_db():
    conn = get_db_connection()
    if not conn:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id VARCHAR(64) PRIMARY KEY,
                    status VARCHAR(32) NOT NULL,
                    progress INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    upload_path TEXT,
                    model_path TEXT,
                    metrics JSONB
                )
            """)
    conn.close()

init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_df_columns(df):
    mapping = {
        'usia': 'Usia',
        'jenis kelamin': 'Jenis Kelamin',
        'riwayat keluarga': 'Riwayat Keluarga',
        'bmi': 'BMI',
        'tekanan darah': 'Tekanan Darah',
        'gula darah': 'Gula Darah',
        'kehamilan': 'Kehamilan',
        'kebiasaan merokok': 'Kebiasaan Merokok',
        'aktivitas fisik': 'Aktivitas Fisik',
        'pola tidur': 'Pola Tidur',
        'diagnosis': 'Diagnosis',
        'class': 'Diagnosis',
        'age': 'Usia',
        'gender': 'Jenis Kelamin',
        'blood pressure': 'Tekanan Darah',
        'bp': 'Tekanan Darah',
        'glucose': 'Gula Darah',
        'smoking': 'Kebiasaan Merokok',
        'physical activity': 'Aktivitas Fisik',
        'sleep pattern': 'Pola Tidur',
    }
    def renamer(c):
        k = c.strip().lower()
        return mapping.get(k, c.strip())
    return df.rename(columns=renamer)

def preprocess_categorical_values(df):
    # Map common binary/categorical values
    replace_map = {
        'yes': 1, 'no': 0,
        'positive': 1, 'negative': 0,
        'laki-laki': 1, 'perempuan': 0,
        'male': 1, 'female': 0,
        'pria': 1, 'wanita': 0,
        'positif': 1, 'negatif': 0,
        'ya': 1, 'tidak': 0
    }
    # Apply to all object columns
    for col in df.select_dtypes(include=['object']).columns:
        # Try to map values (case insensitive)
        try:
            df[col] = df[col].apply(lambda x: replace_map.get(str(x).lower().strip(), x) if isinstance(x, str) else x)
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    return df

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Proses file dan jalankan model
        try:
            results = process_file(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400


def process_file(filepath):
    # Baca data
    try:
        data = pd.read_csv(filepath, sep=None, engine='python')
    except:
        data = pd.read_csv(filepath)
        
    data = normalize_df_columns(data)
    data = preprocess_categorical_values(data)
    
    if 'Diagnosis' not in data.columns:
        raise ValueError("Target column 'Diagnosis' (or alias) not found in dataset")
        
    # Pisahkan fitur dan target
    X = data.drop('Diagnosis', axis=1)
    # Filter only numeric
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features found after preprocessing")
        
    X = X[numeric_features]
    y = data['Diagnosis']
    
    # Inisialisasi model
    rf_model = RandomForestModel()
    svm_model = SVMModel()
    lr_model = LogisticRegressionModel()
    
    # Inisialisasi selector
    rfe_selector = RFESelector()
    bfs_selector = BorutaSelector()
    
    # Hasil untuk semua kombinasi model dan seleksi fitur
    results = {
        'baseline': {},
        'rfe': {},
        'bfs': {}
    }
    
    # Baseline (tanpa seleksi fitur)
    results['baseline']['random_forest'] = evaluate_model(rf_model, X, y)
    results['baseline']['svm'] = evaluate_model(svm_model, X, y)
    results['baseline']['logistic_regression'] = evaluate_model(lr_model, X, y)
    
    # Dengan RFE
    X_rfe = rfe_selector.select_features(X, y)
    results['rfe']['random_forest'] = evaluate_model(rf_model, X_rfe, y)
    results['rfe']['svm'] = evaluate_model(svm_model, X_rfe, y)
    results['rfe']['logistic_regression'] = evaluate_model(lr_model, X_rfe, y)
    results['rfe']['selected_features'] = rfe_selector.get_selected_features()
    
    # Dengan BFS
    X_bfs = bfs_selector.select_features(X, y)
    results['bfs']['random_forest'] = evaluate_model(rf_model, X_bfs, y)
    results['bfs']['svm'] = evaluate_model(svm_model, X_bfs, y)
    results['bfs']['logistic_regression'] = evaluate_model(lr_model, X_bfs, y)
    results['bfs']['selected_features'] = bfs_selector.get_selected_features()
    
    return results


def evaluate_model(model, X, y):
    # Evaluasi model dengan 5-fold cross validation
    metrics = model.evaluate(X, y)
    return metrics


# ===== API Sesi Training Bertahap =====
@app.route('/api/training/session', methods=['POST'])
def create_training_session():
    session_id = str(uuid.uuid4())
    name = None
    try:
        payload = request.get_json(silent=True) or {}
        name = payload.get('name')
    except Exception:
        name = None
    sessions[session_id] = {
        'status': 'created',
        'progress': 0,
        'name': name or 'Training Session',
        'upload_path': None,
        'data': None,
        'X': None,
        'y': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'pipeline': None,
        'best_model': None,
        'metrics': None,
        'model_path': None,
    }
    conn = get_db_connection()
    if conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO training_sessions (id, status, progress, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, 'created', 0, datetime.utcnow(), datetime.utcnow()))
        conn.close()
    return jsonify({'session_id': session_id})


@app.route('/api/training/<session_id>/upload', methods=['POST'])
def training_upload(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
    file.save(filepath)
    sessions[session_id]['upload_path'] = filepath
    sessions[session_id]['status'] = 'uploaded'
    sessions[session_id]['progress'] = 10
    conn = get_db_connection()
    if conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE training_sessions SET status=%s, progress=%s, upload_path=%s, updated_at=%s
                    WHERE id=%s
                """, ('uploaded', 10, filepath, datetime.utcnow(), session_id))
        conn.close()
    return jsonify({'message': 'File uploaded', 'session_id': session_id})


@app.route('/api/training/<session_id>/preprocess', methods=['POST'])
def training_preprocess(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    upload_path = sessions[session_id].get('upload_path')
    if not upload_path:
        return jsonify({'error': 'No data uploaded yet'}), 400
    try:
        try:
            data = pd.read_csv(upload_path, sep=None, engine='python')
        except:
            data = pd.read_csv(upload_path)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
    
    data = normalize_df_columns(data)
    data = preprocess_categorical_values(data)
    
    # Drop rows where Diagnosis is NaN
    if 'Diagnosis' in data.columns:
        data = data.dropna(subset=['Diagnosis'])
        X_all = data.drop('Diagnosis', axis=1)
        y = data['Diagnosis']
        label_present = True
    else:
        X_all = data
        y = None
        label_present = False
        
    # Drop non-feature columns (High Cardinality)
    # Heuristic: Drop object columns with > 90% unique values (likely IDs)
    drop_cols = []
    for col in X_all.select_dtypes(include=['object', 'string']).columns:
        if X_all[col].nunique() > 0.9 * len(X_all):
            drop_cols.append(col)
    
    if drop_cols:
        X_all = X_all.drop(columns=drop_cols)
        
    numeric_features = X_all.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_all.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if not numeric_features and not categorical_features:
        return jsonify({'error': 'No features found.'}), 400
        
    # Store raw data and feature lists for splitting
    sessions[session_id]['data'] = data
    sessions[session_id]['X_raw'] = X_all
    sessions[session_id]['y'] = y
    sessions[session_id]['numeric_features'] = numeric_features
    sessions[session_id]['categorical_features'] = categorical_features
    
    sessions[session_id]['status'] = 'preprocessed'
    sessions[session_id]['progress'] = 30
    
    conn = get_db_connection()
    if conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE training_sessions SET status=%s, progress=%s, updated_at=%s
                    WHERE id=%s
                """, ('preprocessed', 30, datetime.utcnow(), session_id))
        conn.close()
        
    return jsonify({
        'message': 'Preprocessing complete (Cleaning only)', 
        'n_features': X_all.shape[1], 
        'label_present': label_present,
        'note': 'Scaling and Encoding will be performed after Data Split to prevent leakage.'
    })

    # The old pipeline logic is moved to training_split



@app.route('/api/training/<session_id>/split', methods=['POST'])
def training_split(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    # Use X_raw if available (new correct flow), else fallback to X (old flow)
    X = sessions[session_id].get('X_raw')
    if X is None:
        X = sessions[session_id].get('X') # Fallback if someone used old preprocess
        
    y = sessions[session_id].get('y')
    
    if X is None:
        return jsonify({'error': 'Preprocessing not completed'}), 400
    if y is None:
        return jsonify({'error': 'Dataset is unlabeled. Provide Diagnosis column to split.'}), 400
        
    # Split raw data first
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        # Fallback if stratify fails (e.g. rare classes)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Now apply Preprocessing Pipeline (Fit on Train, Transform Train & Test)
    numeric_features = sessions[session_id].get('numeric_features')
    categorical_features = sessions[session_id].get('categorical_features')
    
    # If not in session (fallback), detect again
    if numeric_features is None:
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    from sklearn.compose import ColumnTransformer
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    try:
        # Fit on TRAIN only
        X_train_processed_np = preprocessor.fit_transform(X_train_raw)
        # Transform TEST
        X_test_processed_np = preprocessor.transform(X_test_raw)
        
        # Convert sparse to dense if necessary
        try:
            import scipy
            if scipy.sparse.issparse(X_train_processed_np):
                X_train_processed_np = X_train_processed_np.toarray()
            if scipy.sparse.issparse(X_test_processed_np):
                X_test_processed_np = X_test_processed_np.toarray()
        except Exception:
            try:
                # Fallback using toarray if available
                if hasattr(X_train_processed_np, 'toarray'):
                    X_train_processed_np = X_train_processed_np.toarray()
                if hasattr(X_test_processed_np, 'toarray'):
                    X_test_processed_np = X_test_processed_np.toarray()
            except Exception:
                pass
        
        # Get feature names
        new_cols = []
        new_cols.extend(numeric_features)
        if categorical_features:
            try:
                cat_encoder = preprocessor.named_transformers_['cat']['encoder']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_names = cat_encoder.get_feature_names_out(categorical_features)
                    new_cols.extend(cat_names)
                else:
                    new_cols.extend([f"cat_{i}" for i in range(X_train_processed_np.shape[1] - len(numeric_features))])
            except:
                 new_cols.extend([f"cat_{i}" for i in range(X_train_processed_np.shape[1] - len(numeric_features))])
        
        # Adjust columns length if needed (for passthrough)
        if len(new_cols) != X_train_processed_np.shape[1]:
            if len(new_cols) < X_train_processed_np.shape[1]:
                new_cols.extend([f"feature_{i}" for i in range(len(new_cols), X_train_processed_np.shape[1])])
            else:
                new_cols = new_cols[:X_train_processed_np.shape[1]]

        # Convert back to DataFrames
        X_train = pd.DataFrame(X_train_processed_np, columns=new_cols, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test_processed_np, columns=new_cols, index=X_test_raw.index)
        
        # Store in session
        sessions[session_id]['X_train'] = X_train
        sessions[session_id]['X_test'] = X_test
        sessions[session_id]['y_train'] = y_train
        sessions[session_id]['y_test'] = y_test
        sessions[session_id]['pipeline'] = preprocessor
        
        sessions[session_id]['status'] = 'split'
        sessions[session_id]['progress'] = 50
        
        conn = get_db_connection()
        if conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE training_sessions SET status=%s, progress=%s, updated_at=%s
                        WHERE id=%s
                    """, ('split', 50, datetime.utcnow(), session_id))
            conn.close()
            
        return jsonify({
            'message': 'Data split & Preprocessing complete', 
            'train_size': int(len(y_train)), 
            'test_size': int(len(y_test)),
            'n_features': X_train.shape[1]
        })
        
    except Exception as e:
        return jsonify({'error': f'Split/Preprocess failed: {str(e)}'}), 400



@app.route('/api/training/<session_id>/train', methods=['POST'])
def training_train(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    X_train = sessions[session_id].get('X_train')
    y_train = sessions[session_id].get('y_train')
    if X_train is None or y_train is None:
        return jsonify({'error': 'Data not split yet'}), 400

    # Ensure y_train is integer/numeric for classification
    try:
        y_train = y_train.astype(int)
    except:
        # Try encoding if string
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        sessions[session_id]['label_encoder'] = le

    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get('epochs', 100))
    batch_size = int(payload.get('batch_size', 32))
    learning_rate = float(payload.get('learning_rate', 0.001))

    logs = []
    logs.append(f"Training Config: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}")

    # Define Models
    def get_models():
        rf = RandomForestClassifier(n_estimators=epochs, random_state=42)
        svm = SVC(probability=True, max_iter=epochs if epochs > 0 else -1, random_state=42)
        lr = LogisticRegression(max_iter=epochs, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=epochs, batch_size=batch_size, learning_rate_init=learning_rate, random_state=42, early_stopping=True)
        return [('Random Forest', rf), ('SVM', svm), ('Logistic Regression', lr), ('Neural Network', mlp)]

    best_global_score = -np.inf
    best_global_model = None
    best_global_name = ""
    
    comparison_results = []
    selected_rfe_cols = []
    selected_bfs_cols = []

    # --- Phase 1: Baseline (All Features) ---
    logs.append("\n=== Phase 1: Baseline (All Features) ===")
    models = get_models()
    for name, m in models:
        try:
            start_time = time.time()
            m.fit(X_train, y_train)
            duration = time.time() - start_time
            score = float(m.score(X_train, y_train))
            logs.append(f"[Baseline] {name}: Acc={score:.2%} ({duration:.2f}s)")
            
            comparison_results.append({
                'phase': 'Baseline',
                'model': name,
                'accuracy': score,
                'duration': duration
            })

            if score > best_global_score:
                best_global_score = score
                best_global_model = m
                best_global_name = f"{name} (Baseline)"
                # Clear feature mask for baseline
                if 'feature_mask' in sessions[session_id]:
                    del sessions[session_id]['feature_mask']
        except Exception as e:
            logs.append(f"[Baseline] {name} Failed: {str(e)}")

    # --- Phase 2: RFE Selection ---
    logs.append("\n=== Phase 2: Feature Selection (RFE) ===")
    try:
        rfe = RFESelector()
        # Select top 50% features
        X_train_rfe = rfe.select_features(X_train, y_train)
        selected_rfe_cols = rfe.get_selected_features()
        logs.append(f"RFE selected {len(selected_rfe_cols)} features: {', '.join(selected_rfe_cols[:5])}...")
        
        # Train on RFE data
        models_rfe = get_models()
        for name, m in models_rfe:
            try:
                start_time = time.time()
                m.fit(X_train_rfe, y_train)
                duration = time.time() - start_time
                score = float(m.score(X_train_rfe, y_train))
                logs.append(f"[RFE] {name}: Acc={score:.2%} ({duration:.2f}s)")
                
                comparison_results.append({
                    'phase': 'RFE',
                    'model': name,
                    'accuracy': score,
                    'duration': duration
                })

                if score > best_global_score:
                    best_global_score = score
                    best_global_model = m
                    best_global_name = f"{name} (RFE)"
                    sessions[session_id]['feature_mask'] = ('rfe', selected_rfe_cols)
            except Exception as e:
                logs.append(f"[RFE] {name} Failed: {str(e)}")
    except Exception as e:
        logs.append(f"RFE Selection Failed: {str(e)}")

    # --- Phase 3: Boruta Selection (BFS) ---
    logs.append("\n=== Phase 3: Feature Selection (Boruta) ===")
    try:
        # Boruta can be slow, use with caution on large data
        bfs = BorutaSelector()
        # Boruta needs numpy y
        y_train_np = y_train if hasattr(y_train, 'values') else np.array(y_train)
        
        X_train_bfs = bfs.select_features(X_train, y_train_np)
        selected_bfs_cols = bfs.get_selected_features()
        
        if len(selected_bfs_cols) == 0:
             logs.append("Boruta found no relevant features. Skipping BFS training.")
        else:
            logs.append(f"Boruta selected {len(selected_bfs_cols)} features: {', '.join(selected_bfs_cols[:5])}...")
            
            # Train on BFS data
            models_bfs = get_models()
            for name, m in models_bfs:
                try:
                    start_time = time.time()
                    m.fit(X_train_bfs, y_train)
                    duration = time.time() - start_time
                    score = float(m.score(X_train_bfs, y_train))
                    logs.append(f"[BFS] {name}: Acc={score:.2%} ({duration:.2f}s)")
                    
                    comparison_results.append({
                        'phase': 'Boruta',
                        'model': name,
                        'accuracy': score,
                        'duration': duration
                    })

                    if score > best_global_score:
                        best_global_score = score
                        best_global_model = m
                        best_global_name = f"{name} (Boruta)"
                        sessions[session_id]['feature_mask'] = ('bfs', selected_bfs_cols)
                except Exception as e:
                    logs.append(f"[BFS] {name} Failed: {str(e)}")
    except Exception as e:
        logs.append(f"Boruta Selection Failed: {str(e)}")

    # Finalize
        if best_global_model:
            try:
                headers = ["Phase", "Model", "Accuracy", "Time (s)"]
                rows = []
                for r in comparison_results:
                    rows.append([
                        str(r.get('phase', '')),
                        str(r.get('model', '')),
                        f"{float(r.get('accuracy', 0.0))*100:.2f}%",
                        f"{float(r.get('duration', 0.0)):.3f}"
                    ])
                widths = [len(h) for h in headers]
                for row in rows:
                    for i, val in enumerate(row):
                        if len(val) > widths[i]:
                            widths[i] = len(val)
                def fmt_row(row_vals):
                    return " | ".join(val.ljust(widths[i]) for i, val in enumerate(row_vals))
                line = "-+-".join("-"*w for w in widths)
                table_lines = []
                table_lines.append(fmt_row(headers))
                table_lines.append(line)
                for row in rows:
                    table_lines.append(fmt_row(row))
                table_str = "\n".join(table_lines)
                logs.append("\n=== Summary Table ===")
                logs.append(table_str)
                print("\n=== Summary Table ===")
                print(table_str)

                # Process table (feature counts per phase)
                proc_headers = ["Phase", "Features"]
                base_features = X_train.shape[1] if hasattr(X_train, 'shape') else None
                proc_rows = []
                proc_rows.append(["Baseline", str(base_features) if base_features is not None else "-"])
                proc_rows.append(["RFE", str(len(selected_rfe_cols)) if selected_rfe_cols else "-"])
                proc_rows.append(["Boruta", str(len(selected_bfs_cols)) if selected_bfs_cols else "-"])
                proc_widths = [len(h) for h in proc_headers]
                for row in proc_rows:
                    for i, val in enumerate(row):
                        if len(val) > proc_widths[i]:
                            proc_widths[i] = len(val)
                def fmt_proc(row_vals):
                    return " | ".join(val.ljust(proc_widths[i]) for i, val in enumerate(row_vals))
                proc_line = "-+-".join("-"*w for w in proc_widths)
                proc_table_lines = [fmt_proc(proc_headers), proc_line]
                for row in proc_rows:
                    proc_table_lines.append(fmt_proc(row))
                proc_table_str = "\n".join(proc_table_lines)
                logs.append("\n=== Process Table ===")
                logs.append(proc_table_str)
                print("\n=== Process Table ===")
                print(proc_table_str)
            except Exception as e:
                logs.append(f"Failed to render summary table: {str(e)}")
            sessions[session_id]['best_model'] = (best_global_name, best_global_model)
            sessions[session_id]['status'] = 'trained'
            sessions[session_id]['progress'] = 80
            sessions[session_id]['training_logs'] = logs
            sessions[session_id]['comparison_results'] = comparison_results
            
            return jsonify({
                'message': 'Training & Comparison Complete',
                'best_model': best_global_name,
                'train_score': best_global_score,
                'logs': logs,
                'comparison': comparison_results,
                'process_summary': [
                    {'phase': 'Baseline', 'features': base_features},
                    {'phase': 'RFE', 'features': len(selected_rfe_cols) if selected_rfe_cols else 0},
                    {'phase': 'Boruta', 'features': len(selected_bfs_cols) if selected_bfs_cols else 0}
                ]
            })
    else:
        return jsonify({'error': 'All training attempts failed', 'logs': logs}), 500


@app.route('/api/training/<session_id>/evaluate', methods=['POST'])
def training_evaluate(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    X_test = sessions[session_id].get('X_test')
    y_test = sessions[session_id].get('y_test')
    best = sessions[session_id].get('best_model')
    
    if best is None or X_test is None or y_test is None:
        return jsonify({'error': 'Training not completed'}), 400
    
    name, model = best
    
    # Check if we need to filter features
    feature_mask = sessions[session_id].get('feature_mask')
    if feature_mask:
        mask_type, selected_cols = feature_mask
        # Filter X_test
        # Ensure X_test is DataFrame to select by name
        if isinstance(X_test, pd.DataFrame):
            X_eval = X_test[selected_cols]
        else:
            # Should not happen if we keep X as DF in sessions
            return jsonify({'error': 'X_test format mismatch for feature selection'}), 500
    else:
        X_eval = X_test

    try:
        y_pred = model.predict(X_eval)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_eval)[:, 1]
            except:
                pass
                
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        sessions[session_id]['metrics'] = {'model': name, **metrics}
        sessions[session_id]['status'] = 'evaluated'
        sessions[session_id]['progress'] = 85
        
        conn = get_db_connection()
        if conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE training_sessions SET status=%s, progress=%s, metrics=%s::jsonb, updated_at=%s
                        WHERE id=%s
                    """, ('evaluated', 85, json.dumps(sessions[session_id]['metrics']), datetime.utcnow(), session_id))
            conn.close()
            
        return jsonify({'message': 'Evaluation complete', 'metrics': sessions[session_id]['metrics']})
    except Exception as e:
        return jsonify({'error': f"Evaluation failed: {str(e)}"}), 500


@app.route('/api/training/<session_id>/save', methods=['POST'])
def training_save(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    best = sessions[session_id].get('best_model')
    pipeline = sessions[session_id].get('pipeline')
    if best is None or pipeline is None:
        return jsonify({'error': 'No trained model to save'}), 400
    
    # Get feature lists to ensure prediction consistency
    numeric_features = sessions[session_id].get('numeric_features', [])
    categorical_features = sessions[session_id].get('categorical_features', [])
    feature_mask = sessions[session_id].get('feature_mask') # (type, selected_cols)
    
    name, model = best
    bundle = {
        'model_name': name, 
        'model': model, 
        'pipeline': pipeline,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_mask': feature_mask
    }
    
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{session_id}.joblib")
    joblib.dump(bundle, model_path)
    sessions[session_id]['model_path'] = model_path
    sessions[session_id]['status'] = 'saved'
    sessions[session_id]['progress'] = 100
    conn = get_db_connection()
    if conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE training_sessions SET status=%s, progress=%s, model_path=%s, updated_at=%s
                    WHERE id=%s
                """, ('saved', 100, model_path, datetime.utcnow(), session_id))
        conn.close()
    return jsonify({'message': 'Model saved', 'model_path': model_path})



@app.route('/api/training/<session_id>/status', methods=['GET'])
def training_status(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'status': s['status'], 'progress': s['progress'], 'name': s.get('name')})

@app.route('/api/training/<session_id>', methods=['GET'])
def training_details(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'id': session_id,
        'name': s.get('name'),
        'status': s.get('status'),
        'progress': s.get('progress'),
        'upload_path': s.get('upload_path'),
        'model_path': s.get('model_path'),
        'metrics': s.get('metrics')
    })

@app.route('/api/training/sessions', methods=['GET'])
def list_sessions():
    return jsonify([
        {
            'id': sid,
            'name': s.get('name'),
            'status': s.get('status'),
            'progress': s.get('progress'),
            'upload_path': s.get('upload_path'),
            'model_path': s.get('model_path')
        }
        for sid, s in sessions.items()
    ])

@app.route('/api/training/<session_id>/name', methods=['PATCH'])
def rename_session(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    payload = request.get_json(silent=True) or {}
    new_name = payload.get('name')
    if not new_name:
        return jsonify({'error': 'Name is required'}), 400
    s['name'] = new_name
    return jsonify({'message': 'Name updated', 'name': new_name})

@app.route('/api/training/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    up = s.get('upload_path')
    mp = s.get('model_path')
    try:
        if up and os.path.exists(up):
            os.remove(up)
    except Exception:
        pass
    try:
        if mp and os.path.exists(mp):
            os.remove(mp)
    except Exception:
        pass
    del sessions[session_id]
    return jsonify({'message': 'Session deleted'})


# ===== API Live Prediction =====
@app.route('/api/predict/live', methods=['POST'])
def predict_live():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    model_id = request.form.get('model_id')
    model_path = None
    if model_id:
        candidate = os.path.join(app.config['MODEL_FOLDER'], f"{model_id}.joblib")
        if os.path.exists(candidate):
            model_path = candidate
    if not model_path:
        files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.joblib')]
        if not files:
            return jsonify({'error': 'No saved model found'}), 400
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['MODEL_FOLDER'], x)), reverse=True)
        model_path = os.path.join(app.config['MODEL_FOLDER'], files[0])
        
    bundle = joblib.load(model_path)
    pipeline = bundle['pipeline']
    model = bundle['model']
    numeric_features = bundle.get('numeric_features', [])
    categorical_features = bundle.get('categorical_features', [])
    feature_mask = bundle.get('feature_mask')
    
    try:
        df = pd.read_csv(file, sep=None, engine='python')
    except:
        file.seek(0)
        df = pd.read_csv(file)
        
    df = normalize_df_columns(df)
    df = preprocess_categorical_values(df)
    
    if 'Diagnosis' in df.columns:
        X = df.drop('Diagnosis', axis=1)
        y_true = df['Diagnosis']
    else:
        X = df
        y_true = None
        
    # Drop rows with NaN label to ensure valid evaluation
    if y_true is not None:
        try:
            mask = ~pd.isna(y_true)
            X = X[mask]
            y_true = y_true[mask]
        except Exception:
            pass
        
    # Filter features to match training
    # If we have stored features, use them
    if numeric_features or categorical_features:
        # Check if all required features are present
        missing_cols = []
        for col in numeric_features + categorical_features:
            if col not in X.columns:
                missing_cols.append(col)
        
        if missing_cols:
             return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400
             
        # Select only relevant columns to avoid passthrough of garbage
        X = X[numeric_features + categorical_features]
    else:
        # Legacy fallback (might fail if new pipeline is used)
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_features]
    
    try:
        Xp = pipeline.transform(X)
        
        # Apply feature selection mask if exists
        if feature_mask:
            mask_type, selected_cols = feature_mask
            
            # Xp is numpy array or sparse matrix from pipeline
            # If pipeline returns DataFrame (not standard sklearn yet), we could use names
            # But standard Pipeline returns numpy array.
            # We need to reconstruct DF or apply mask on array?
            # RFE/Boruta select_features returned a DataFrame or numpy array?
            # In training_train:
            # X_train_rfe = rfe.select_features(X_train, y_train)
            # X_train was DataFrame (from split). select_features returns DataFrame (usually, if implemented to support it).
            
            # Let's check RFE/Boruta implementation.
            # But here Xp is output of pipeline.transform(X).
            # If pipeline ends with passthrough, Xp is numpy array.
            
            # If feature_mask has selected_cols (names), we need column names of Xp.
            # Reconstructing names:
            new_cols = []
            new_cols.extend(numeric_features)
            if categorical_features:
                try:
                    cat_encoder = pipeline.named_transformers_['cat']['encoder']
                    if hasattr(cat_encoder, 'get_feature_names_out'):
                        cat_names = cat_encoder.get_feature_names_out(categorical_features)
                        new_cols.extend(cat_names)
                    else:
                        new_cols.extend([f"cat_{i}" for i in range(Xp.shape[1] - len(numeric_features))])
                except:
                    new_cols.extend([f"cat_{i}" for i in range(Xp.shape[1] - len(numeric_features))])
            
            if len(new_cols) != Xp.shape[1]:
                # Fallback
                 new_cols = [f"f{i}" for i in range(Xp.shape[1])]

            # Create DF to select by name
            Xp_df = pd.DataFrame(Xp, columns=new_cols)
            
            # Ensure selected_cols exist
            available_cols = [c for c in selected_cols if c in Xp_df.columns]
            if len(available_cols) < len(selected_cols):
                # Warning or Error?
                pass
                
            X_final = Xp_df[available_cols]
        else:
            X_final = Xp
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed (preprocessing): {str(e)}'}), 400
        
    try:
        y_pred = model.predict(X_final)
        y_prob = model.predict_proba(X_final)[:, 1] if hasattr(model, 'predict_proba') else None
        response = {'predictions': y_pred.tolist()}
        if y_true is not None:
            response['metrics'] = calculate_metrics(y_true, y_pred, y_prob)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Prediction failed (model inference): {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True)

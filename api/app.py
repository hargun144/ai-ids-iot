from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import functools
import os
import json

app = Flask(__name__)

API_KEY = "priya_ids_iot"

def require_key(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapper

MODEL_DIR = "models"

_dl_models = {}
EXPECTED_SHAPES = {'unsw': (20, 42)}

def get_dl_model(dataset):
    if dataset not in _dl_models:
        import tensorflow as tf
        path_map = {'unsw': "priya_best_model_unsw_cnn.keras"}
        if dataset not in path_map:
            return None
        _dl_models[dataset] = tf.keras.models.load_model(os.path.join(MODEL_DIR, path_map[dataset]))
    return _dl_models[dataset]

ML_MODEL_FILES = {
    "unsw": "unsw_best_ml_xgboost.pkl",
    "ton": "ton_best_ml_random_forest.pkl",
}
_ml_loaded = {}

def get_ml_bundle(dataset):
    if dataset not in _ml_loaded:
        if dataset == "nbaiot":
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(os.path.join(MODEL_DIR, "nbaiot_best_ml_xgboost.json"))
            with open(os.path.join(MODEL_DIR, "nbaiot_best_ml_xgboost_features.json")) as f:
                features = json.load(f)
            _ml_loaded[dataset] = {"model": model, "features": features}
        elif dataset in ML_MODEL_FILES:
            path = os.path.join(MODEL_DIR, ML_MODEL_FILES[dataset])
            with open(path, "rb") as f:
                _ml_loaded[dataset] = pickle.load(f)
        else:
            return None
    return _ml_loaded[dataset]


@app.route('/predict', methods=['POST'])
@require_key
def predict():
    data = request.get_json()
    model_type = data.get('model', 'ml')
    dataset = data.get('dataset')

    if model_type == 'dl':
        model = get_dl_model(dataset)
        if model is None:
            return jsonify({'error': f"Unknown DL dataset '{dataset}'"}), 400
        sequence = np.array(data.get('sequence'), dtype=np.float32)
        expected_shape = EXPECTED_SHAPES[dataset]
        if sequence.shape != expected_shape:
            return jsonify({'error': f'Expected shape {expected_shape}, got {sequence.shape}'}), 400
        sequence = sequence.reshape(1, *expected_shape)
        prob = float(model.predict(sequence, verbose=0)[0][0])
        label = 'Attack' if prob > 0.5 else 'Normal'
        return jsonify({'model': 'DL-1D-CNN', 'dataset': dataset, 'prediction': label, 'confidence': round(prob, 4)})

    else:
        bundle = get_ml_bundle(dataset)
        if bundle is None:
            return jsonify({'error': f"Unknown ML dataset '{dataset}'"}), 400
        model, features = bundle['model'], bundle['features']
        req_features = data.get('features', {})
        missing = [f for f in features if f not in req_features]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 422
        row = pd.DataFrame([{f: req_features[f] for f in features}])
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0].tolist() if hasattr(model, "predict_proba") else None
        label = "Attack Detected" if int(pred) == 1 else "Normal"
        return jsonify({'model': type(model).__name__, 'dataset': dataset, 'prediction': label,
                         'raw_class': int(pred), 'probability': proba})


if __name__ == '__main__':
    app.run(port=8000)
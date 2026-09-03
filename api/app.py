from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import functools

app = Flask(__name__)

dl_models = {
    'unsw': tf.keras.models.load_model("models/priya_best_model_unsw_cnn.keras")
}

EXPECTED_SHAPES = {
    'unsw': (20, 42),
}

API_KEY = "priya_ids_iot"

def require_key(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapper

@app.route('/predict', methods=['POST'])
@require_key
def predict():
    data = request.get_json()
    dataset = data.get('dataset')
    sequence = np.array(data.get('sequence'), dtype=np.float32)

    if dataset not in dl_models:
        return jsonify({'error': f"Unknown dataset '{dataset}'. Currently supported: {list(dl_models.keys())}"}), 400

    expected_shape = EXPECTED_SHAPES[dataset]
    if sequence.shape != expected_shape:
        return jsonify({'error': f'Expected shape {expected_shape}, got {sequence.shape}'}), 400

    sequence = sequence.reshape(1, *expected_shape)
    prob = float(dl_models[dataset].predict(sequence, verbose=0)[0][0])
    label = 'Attack' if prob > 0.5 else 'Normal'

    return jsonify({'model': 'DL-1D-CNN', 'dataset': dataset, 'prediction': label, 'confidence': round(prob, 4)})

if __name__ == '__main__':
    app.run(port=8000)

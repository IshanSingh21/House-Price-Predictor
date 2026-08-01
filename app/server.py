"""
Flask API Server for House Price Prediction
Serves the web UI and handles prediction requests.
"""

import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'house_price_model.pkl')

model_data = None

def load_model():
    global model_data
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Run 'python model/train.py' first.")
        sys.exit(1)
    model_data = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully")
    print(f"   Features: {model_data['feature_names']}")


@app.route('/')
def index():
    """Serve the main prediction UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()

        # Extract features in the correct order
        feature_values = []
        for feat in model_data['feature_names']:
            value = data.get(feat)
            if value is None:
                return jsonify({'error': f'Missing feature: {feat}'}), 400
            feature_values.append(float(value))

        # Scale and predict
        features_array = np.array(feature_values).reshape(1, -1)
        features_scaled = model_data['scaler'].transform(features_array)
        prediction = model_data['model'].predict(features_scaled)[0]

        # Get feature importance if available
        importance = {}
        if hasattr(model_data['model'], 'feature_importances_'):
            for name, imp in zip(model_data['feature_names'], model_data['model'].feature_importances_):
                importance[name] = round(float(imp), 4)

        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'feature_importance': importance
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model_data is not None})


if __name__ == '__main__':
    load_model()
    print("\n🌐 Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

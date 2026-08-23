from flask import Flask, request, jsonify, render_template
import mlflow
import pandas as pd
import os
import logging
from mlflow.pyfunc import load_model
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('AUTH_USERNAME', 'admin')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('AUTH_PASSWORD', 'password')

basic_auth = BasicAuth(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables and default values
MODEL_URI = os.getenv('MODEL_URI', 'models:/fraud_detection/Production')
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Load the model
try:
    model = load_model(MODEL_URI)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    """Endpoint to make fraud detection predictions."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.form.to_dict()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Input validation
        required_fields = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields in input data'}), 400

        df = pd.DataFrame([data])
        prediction = float(model.predict(df)[0])  # Probability of class 1 (fraud)
        logging.info(f"Prediction: {prediction}")
        is_fraud = bool(prediction > 0.5)

        # Log prediction and input data for monitoring
        logging.info(f"Input data: {data}")
        logging.info(f"Prediction: {prediction}, Is Fraud: {is_fraud}")

        return jsonify({'prediction': prediction, 'is_fraud': is_fraud})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(SERVER_PORT), debug=DEBUG_MODE)
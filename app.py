from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("fraud_detection_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
optimal_threshold = joblib.load("optimal_threshold.pkl")

# Home route (renders UI)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.json

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess input
        preprocessed_data = preprocessor.transform(input_data)

        # Predict fraud probability
        fraud_probability = model.predict_proba(preprocessed_data)[:, 1][0]

        # Classify fraud using optimal threshold
        is_fraud = int(fraud_probability >= optimal_threshold)

        return jsonify({
            "fraud_probability": round(fraud_probability, 4),
            "is_fraud": is_fraud
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

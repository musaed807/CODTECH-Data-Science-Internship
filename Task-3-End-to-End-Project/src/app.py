from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = os.path.join("..", "model", "loan_approval_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Loan Approval Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON format:
    {
        "Age": 35,
        "Income": 50000,
        "Credit_Score": 720,
        "Loan_Amount": 200000,
        "Loan_Term": 360,
        "Employment_Status": "Employed"
    }
    """

    data = request.get_json()

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)

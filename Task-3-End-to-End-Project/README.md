# Task 3: Loan Approval Prediction API

## Project Overview
This project demonstrates a complete **end-to-end data science workflow**
as part of the CODTECH Data Science Internship.

It covers:
- Data preprocessing
- Machine learning model training
- API deployment using Flask

The API predicts whether a loan application is **Approved** or **Rejected**
based on applicant details.

---

## Problem Statement
Financial institutions must evaluate loan applications efficiently.
This project builds a predictive model and exposes it via a REST API
to simulate real-world decision automation.

---

## Dataset
Loan application dataset containing features such as:
- Age
- Income
- Credit Score
- Loan Amount
- Loan Term
- Employment Status

Target variable:
- Loan Approval (binary classification)

---

## Model
- Algorithm: Logistic Regression
- Preprocessing:
  - Missing value imputation
  - One-hot encoding for categorical variables
  - Feature scaling
- Performance:
  - Train Accuracy: ~92%
  - Test Accuracy: ~91%

---

## API Details

### Base URL
http://127.0.0.1:5000/


### Endpoints

#### Health Check


GET /


Response:
```json
{
  "message": "Loan Approval Prediction API is running"
}

Prediction
POST /predict


Sample Request:

{
  "Age": 30,
  "Income": 60000,
  "Credit_Score": 750,
  "Loan_Amount": 150000,
  "Loan_Term": 240,
  "Employment_Status": "Employed"
}


Sample Response:

{
  "prediction": "Approved"
}

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Flask

Joblib

How to Run
Train Model
cd Task-3-End-to-End-Project/src
python train_model.py

Run API
python app.py
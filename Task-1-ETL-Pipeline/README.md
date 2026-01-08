# Task 1: Data Preprocessing ETL Pipeline

## Project Overview
This project implements an automated **ETL (Extract, Transform, Load) pipeline**
for a loan approval dataset as part of the CODTECH Data Science Internship.

The pipeline handles real-world data issues such as missing values,
categorical variables, feature scaling, and dataset splitting.

---

## Dataset Description
The dataset contains customer and loan-related attributes, including:

- Age
- Income
- Credit Score
- Loan Amount
- Loan Term
- Employment Status
- Loan Approval (Target Variable)

---

## ETL Pipeline Steps

### 1. Extract
- Load raw CSV data using Pandas

### 2. Transform
- Drop unnecessary ID columns (if present)
- Automatically detect target column
- Handle missing values:
  - Numerical → Median Imputation
  - Categorical → Most Frequent
- Encode categorical variables using One-Hot Encoding
- Scale numerical features using StandardScaler

### 3. Load
- Split data into training and testing sets
- Save processed feature matrices and labels to disk

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## How to Run

1. Activate virtual environment
2. Navigate to source directory:
   ```bash
   cd Task-1-ETL-Pipeline/src

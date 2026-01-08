import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def main():
    # -----------------------------
    # 1. Load Dataset
    # -----------------------------
    DATA_PATH = os.path.join("..", "data", "loan_prediction.csv")
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # -----------------------------
    # 2. Drop ID Column (if exists)
    # -----------------------------
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)
        print("Loan_ID column dropped")
    else:
        print("Loan_ID column not found — skipping drop")

    # -----------------------------
    # 3. Identify Target Column
    # -----------------------------
    if "Loan_Status" in df.columns:
        target_col = "Loan_Status"
    elif "Loan_Approved" in df.columns:
        target_col = "Loan_Approved"
    else:
        raise ValueError("No valid target column found in dataset")

    print(f"Using target column: {target_col}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # -----------------------------
    # 4. Identify Column Types
    # -----------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)

    # -----------------------------
    # 5. Define Preprocessing Pipelines
    # -----------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    # -----------------------------
    # 6. Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 7. Apply Preprocessing
    # -----------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing completed")

    # -----------------------------
    # 8. Save Processed Data
    # -----------------------------
    output_dir = os.path.join("..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)

    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("Processed data saved successfully")
    print("ETL pipeline execution completed successfully")


if __name__ == "__main__":
    main()

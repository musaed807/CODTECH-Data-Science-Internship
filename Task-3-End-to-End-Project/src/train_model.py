import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def main():
    # -----------------------------
    # 1. Load Dataset
    # -----------------------------
    DATA_PATH = os.path.join("..", "data", "loan_prediction.csv")
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # -----------------------------
    # 2. Identify Target Column
    # -----------------------------
    if "Loan_Status" in df.columns:
        target_col = "Loan_Status"
    elif "Loan_Approved" in df.columns:
        target_col = "Loan_Approved"
    else:
        raise ValueError("No valid target column found")

    print(f"Using target column: {target_col}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # -----------------------------
    # 3. Identify Column Types
    # -----------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # -----------------------------
    # 4. Preprocessing Pipelines
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
    # 5. Full ML Pipeline
    # -----------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # -----------------------------
    # 6. Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 7. Train Model
    # -----------------------------
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # -----------------------------
    # 8. Save Model
    # -----------------------------
    model_dir = os.path.join("..", "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "loan_approval_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

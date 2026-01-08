import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # -----------------------------
    # 1. Load Dataset
    # -----------------------------
    DATA_PATH = os.path.join("..", "data", "spam.csv")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    # Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean text
    df["clean_message"] = df["message"].apply(clean_text)

    # -----------------------------
    # 2. Tokenization & Padding
    # -----------------------------
    MAX_VOCAB_SIZE = 5000
    MAX_SEQUENCE_LENGTH = 50

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_message"])

    sequences = tokenizer.texts_to_sequences(df["clean_message"])
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    X = padded_sequences
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 3. Build Deep Learning Model
    # -----------------------------
    model = Sequential([
        Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=64, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # -----------------------------
    # 4. Train Model
    # -----------------------------
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # -----------------------------
    # 5. Evaluate Model
    # -----------------------------
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # -----------------------------
    # 6. Plot Training History
    # -----------------------------
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # -----------------------------
    # 7. Save Model
    # -----------------------------
    model.save("spam_classifier_model.h5")
    print("Model saved as spam_classifier_model.h5")


if __name__ == "__main__":
    main()

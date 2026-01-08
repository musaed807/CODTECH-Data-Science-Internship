# Task 2: SMS Spam Detection Using Deep Learning

## Project Overview
This project implements a **deep learning-based NLP model**
to classify SMS messages as **Spam** or **Ham** as part of the
CODTECH Data Science Internship.

The solution uses an LSTM neural network to learn patterns
in text data and achieve high classification accuracy.

---

## Dataset
**SMS Spam Collection Dataset**

- Label column: `v1` (spam / ham)
- Text column: `v2` (SMS message)

The dataset contains 5,572 messages and reflects real-world
class imbalance commonly found in spam detection problems.

---

## Approach

### Text Preprocessing
- Lowercasing text
- Removing punctuation and special characters
- Tokenization
- Padding sequences to fixed length

### Model Architecture
- Embedding Layer
- LSTM Layer
- Dense Output Layer (Sigmoid)

---

## Model Performance
- **Test Accuracy:** ~97%
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

The model demonstrates strong generalization and reliable
classification performance.

---

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib

---

## How to Run
```bash
cd Task-2-Deep-Learning/src
python spam_classifier.py

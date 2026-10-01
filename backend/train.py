"""Train the Logistic Regression model and save model + metrics.
Run from project root:  python backend/train.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "student_course_purchase.csv"
MODEL_DIR = ROOT / "backend" / "model"
TARGET = "purchased_course"


def train():
    df = pd.read_csv(DATA_FILE)
    X, y = df.drop(TARGET, axis=1), df[TARGET]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "features": features,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Model trained and saved. Accuracy:", metrics["accuracy"])
    return model, metrics


if __name__ == "__main__":
    train()

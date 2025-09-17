# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import preprocess_texts
import json
import matplotlib.pyplot as plt
import seaborn as sns
# import os

MODEL_PATH = "model/saved_model.joblib"
VECT_PATH = "model/vectorizer.joblib"
DATA_PATH = "data/comments.csv"

def load_model():
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    return model, vect

def evaluate():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text","label"])
    texts = preprocess_texts(df["text"].tolist())
    y_true = df["label"].astype(str).tolist()

    model, vect = load_model()
    X = vect.transform(texts)
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # os.makedirs("model", exist_ok=True)

    # full report
    with open("model/eval_report.json", "w") as f:
        json.dump({"report": report, "confusion_matrix": cm}, f, indent=2)

    # summary
    summary = {
        "accuracy": report["accuracy"],
        "precision_judi": report["judi"]["precision"],
        "recall_judi": report["judi"]["recall"],
        "f1_judi": report["judi"]["f1-score"]
    }
    with open("model/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # confusion matrix heatmap
    labels = sorted(list(set(y_true)))
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("model/confusion_matrix.png")
    plt.close()

    print("Saved evaluation reports & confusion matrix.")
    return summary

if __name__ == "__main__":
    evaluate()

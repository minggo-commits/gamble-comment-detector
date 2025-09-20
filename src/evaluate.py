import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import preprocess_texts
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    cm_array = confusion_matrix(y_true, y_pred)
    cm = cm_array.tolist()

    os.makedirs("model", exist_ok=True)

    # full report
    with open("model/eval_report.json", "w") as f:
        json.dump({"report": report, "confusion_matrix": cm}, f, indent=2)

    # pilih label target dinamis
    target_label = "judi" if "judi" in report else list(report.keys())[0]
    summary = {
        "accuracy": report["accuracy"],
        f"precision_{target_label}": report[target_label]["precision"],
        f"recall_{target_label}": report[target_label]["recall"],
        f"f1_{target_label}": report[target_label]["f1-score"]
    }
    with open("model/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # confusion matrix heatmap
    labels = sorted(list(set(y_true)))
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
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

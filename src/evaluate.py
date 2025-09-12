# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import preprocess_texts
import json

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

    with open("model/eval_report.json", "w") as f:
        json.dump({"report": report, "confusion_matrix": cm}, f, indent=2)

    print("Saved evaluation to model/eval_report.json")
    return report

if __name__ == "__main__":
    evaluate()

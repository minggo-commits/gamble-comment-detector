# src/inference.py
import joblib
from src.preprocessing import clean_text
import os

MODEL_PATH = "model/saved_model.joblib"
VECT_PATH = "model/vectorizer.joblib"

def load():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        raise FileNotFoundError("Model or vectorizer not found. Run src/train.py first.")
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    return model, vect

def predict(text: str):
    model, vect = load()
    text_clean = clean_text(text)
    X = vect.transform([text_clean])
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()[0]
    return {"text": text, "label": pred, "probability": proba}

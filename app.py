# app.py
import gradio as gr
import joblib
import json
import os
from src.preprocessing import clean_text

model = joblib.load("model/saved_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

def predict(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        proba = { "Aman": round(proba[0]*100, 2), "Judi": round(proba[1]*100, 2) }

    label = "🚫 Judi Online" if pred == "judi" or pred == 1 else "✅ Aman"
    return {"Prediksi": label, "Probabilitas (%)": proba}

def load_metrics():
    path = "model/metrics_summary.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

with gr.Blocks() as demo:
    with gr.Tab("Inference"):
        inp = gr.Textbox(lines=3, placeholder="Masukkan komentar...")
        out = gr.JSON()
        gr.Button("Prediksi").click(fn=predict, inputs=inp, outputs=out)

    with gr.Tab("Monitoring"):
        metrics = load_metrics()
        if metrics:
            gr.Markdown("### 📊 Model Performance")
            gr.JSON(metrics)
        if os.path.exists("model/confusion_matrix.png"):
            gr.Image("model/confusion_matrix.png", label="Confusion Matrix")

if __name__ == "__main__":
    demo.launch()

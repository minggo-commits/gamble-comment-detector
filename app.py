# app.py
import gradio as gr
import joblib
import json
from src.preprocessing import clean_text

model = joblib.load("model/saved_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

def predict(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return "🚫 Judi Online" if pred == "judi" else "✅ Aman"

def load_metrics():
    try:
        with open("model/metrics_summary.json") as f:
            return json.load(f)
    except:
        return {"error": "Metrics not available. Run evaluation first."}

with gr.Blocks() as demo:
    with gr.Tab("Inference"):
        gr.Interface(
            fn=predict,
            inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar..."),
            outputs="text",
            title="Deteksi Komentar Judi Online"
        )

    with gr.Tab("Monitoring"):
        gr.Markdown("### 📊 Model Performance Metrics")
        metrics = load_metrics()
        if "error" in metrics:
            gr.Label(metrics)
        else:
            gr.Label({k: round(v, 4) for k, v in metrics.items()})
            gr.Image("model/confusion_matrix.png", label="Confusion Matrix")

if __name__ == "__main__":
    demo.launch()

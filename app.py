# app.py
import gradio as gr
import joblib

# load model & vectorizer
model = joblib.load("saved_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return f"Prediksi: {pred}"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Tulis komentar di sini..."),
    outputs="text",
    title="Judi Comment Detector",
    description="Deteksi apakah komentar mengandung indikasi judi online."
)

if __name__ == "__main__":
    iface.launch()

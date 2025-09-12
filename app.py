import gradio as gr
import joblib
from src.preprocessing import clean_text

model = joblib.load("model/saved_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

def predict(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return "🚫 Judi Online" if pred == 1 else "✅ Aman"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar..."),
    outputs="text",
    title="Deteksi Komentar Judi Online"
)

if __name__ == "__main__":
    demo.launch()

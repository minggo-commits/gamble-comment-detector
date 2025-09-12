# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import predict, load
import uvicorn

class PredictRequest(BaseModel):
    text: str

app = FastAPI(title="Judi Comment Detector - API")

# warm load model at startup
try:
    model, vect = load()
except Exception as e:
    model = None
    vect = None

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    if model is None or vect is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    result = predict(req.text)
    return result

@app.post("/predict_batch")
def predict_batch(texts: list[str]):
    if model is None or vect is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    from src.preprocessing import clean_text
    cleaned = [clean_text(t) for t in texts]
    X = vect.transform(cleaned)
    labels = model.predict(X).tolist()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()
    return {"predictions": labels, "probabilities": proba}

# run with: uvicorn app.main:app --host 0.0.0.0 --port 7860
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)

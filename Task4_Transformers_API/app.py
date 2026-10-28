"""FastAPI inference API. Fine-tune first with train_model.py to create saved_model/."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_DIR=os.getenv("MODEL_DIR","saved_model")
app=FastAPI(title="Week 3 Feedback Classifier", version="1.0.0")
classifier=None

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer feedback text")

@app.on_event("startup")
def load_model_once():
    global classifier
    if not os.path.isdir(MODEL_DIR) or not os.path.isfile(os.path.join(MODEL_DIR,"config.json")):
        raise RuntimeError(f"Model not found at '{MODEL_DIR}'. Run train_model.py first.")
    classifier=pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=-1)

@app.get("/")
def home():
    return {"message":"Feedback classifier API is running","docs":"/docs"}

@app.post("/predict")
def predict(request: PredictRequest):
    text=request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    result=classifier(text, truncation=True)[0]
    return {"text":text,"label":result["label"],"confidence":round(float(result["score"]),4)}

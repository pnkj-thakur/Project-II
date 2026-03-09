import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "model_artifacts"
MAX_LEN = 512


ID2LABEL = {0: "REAL", 1: "FAKE"}

app = FastAPI(title="Fake News Detector")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


class TextIn(BaseModel):
    text: str


@app.post("/predict")
def predict(req: TextIn):
    text = (req.text or "").strip()
    if not text:
        return {"error": "Empty input. Please paste some text."}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())

    return {
        "label": ID2LABEL.get(pred_id, str(pred_id)),
        "prob_real": float(probs[0].item()),
        "prob_fake": float(probs[1].item()),
    }
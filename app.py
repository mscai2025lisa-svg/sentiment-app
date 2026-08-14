from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# EDIT THIS LINE: put your Hugging Face model repo name here
# Example: "yourusername/sentiment-distilbert-imdb"
# ============================================================
MODEL_NAME = "your-hf-username/sentiment-distilbert-imdb"

app = FastAPI(title="Sentiment Analysis API")

# ---- Load model + tokenizer once, when the server starts ----
print("Loading model... this happens once at startup")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL_NAME = "lisaluis717/sentiment-distilbert-imdb"
model.eval()
print("Model loaded successfully!")

LABELS = {0: "Negative", 1: "Positive"}


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1))
        confidence = float(probs[0][pred_id])

    return {
        "text": input.text,
        "label": LABELS[pred_id],
        "confidence": round(confidence, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Serve the webpage (index.html) at the root URL ----
app.mount("/", StaticFiles(directory="static", html=True), name="static")

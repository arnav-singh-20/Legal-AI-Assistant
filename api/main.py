from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.model_utils import LawgorithmPredictor

app = FastAPI(
    title="Legal AI Assistant API",
    description="Legal Query Classification using BERT — classifies Indian legal queries by domain and intent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    try:
        predictor = LawgorithmPredictor()
        print("Model ready!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        predictor = None

class QueryRequest(BaseModel):
    query: str
    class Config:
        json_schema_extra = {
            "example": {
                "query": "My landlord is refusing to return my security deposit after 2 years"
            }
        }

class PredictionResponse(BaseModel):
    query:          str
    domain:         str
    domain_label:   str
    domain_scores:  dict
    intent:         str
    intent_scores:  dict
    message:        str
    action:         str
    urgency_color:  str
    confidence:     float

@app.get("/")
def root():
    return {
        "name":        "Legal AI Assistant",
        "description": "Legal Query Classification API for Indian Legal System",
        "version":     "1.0.0",
        "endpoints": {
            "POST /predict": "Classify a legal query",
            "GET  /health":  "Health check",
            "GET  /docs":    "Interactive API docs (Swagger UI)"
        },
        "domains": ["criminal", "civil", "property", "family", "tax"],
        "intents": ["needs_lawyer", "self_solvable", "urgent", "general_info"]
    }

@app.get("/health")
def health():
    return {
        "status":      "healthy",
        "model_ready": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = predictor.predict(request.query)
    return result

@app.post("/predict/batch")
def predict_batch(queries: list[str]):
    """Classify multiple queries at once"""
    if len(queries) > 10:
        raise HTTPException(status_code=400, detail="Max 10 queries per batch")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    results = [predictor.predict(q) for q in queries]
    return {"results": results, "count": len(results)}

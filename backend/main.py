"""FastAPI backend: serves the prediction API and the frontend.
Run from project root:  uvicorn backend.main:app --reload
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.train import DATA_FILE, MODEL_DIR, TARGET, train

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

# Train automatically if the model file does not exist yet
if not (MODEL_DIR / "model.pkl").exists():
    train()
with open(MODEL_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(MODEL_DIR / "metrics.json") as f:
    metrics = json.load(f)
FEATURES = metrics["features"]

app = FastAPI(title="Course Purchase Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Student(BaseModel):
    age: int = Field(..., ge=10, le=60)
    study_hours_per_week: int = Field(..., ge=0, le=50)
    previous_courses_completed: int = Field(..., ge=0, le=20)
    platform_visits_per_month: int = Field(..., ge=0, le=100)
    assignment_completion_rate: int = Field(..., ge=0, le=100)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(student: Student):
    # DataFrame keeps the column order identical to training
    row = pd.DataFrame([student.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0][1])
    return {
        "purchase": int(prob >= 0.5),
        "probability": round(prob, 4),
        "label": "Likely to purchase" if prob >= 0.5 else "Not likely to purchase",
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.get("/api/insights")
def insights():
    df = pd.read_csv(DATA_FILE)
    corr = df.corr()[TARGET].drop(TARGET).round(3).to_dict()
    means = df.groupby(TARGET).mean().round(1).to_dict(orient="index")
    return {
        "students": len(df),
        "purchase_rate": round(float(df[TARGET].mean()), 3),
        "correlation": corr,
        "mean_non_buyers": means[0],
        "mean_buyers": means[1],
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

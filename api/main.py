
import json
import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "champion.joblib"


# Load the trained model
model = joblib.load(MODEL_PATH)


# JSON logger
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_data)


logger = logging.getLogger("mp136_api")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


# FastAPI application
app = FastAPI(
    title="MP136 Parcel Delivery Prediction API",
    description="Predicts parcel delivery time in days.",
    version="1.0.0",
)


# Request validation
class PredictionRequest(BaseModel):
    distance_km: float
    weight_kg: float
    hub_backlog: float = Field(le=9000)
    pin_code_tier: int
    declared_value_rs: float
    transfers: int = Field(ge=0)
    service: str
    destination: str


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert request into model input
    input_data = request.model_dump()

    # The trained model expects date_ordinal,
    # but the API specification does not accept a date.
    # NaN lets the trained median imputer handle it.
    input_data["date_ordinal"] = math.nan

    # Create DataFrame in the model's expected column order
    features = pd.DataFrame([input_data])[
        model.feature_names_in_
    ]

    # Generate prediction
    prediction = float(model.predict(features)[0])

    # Log prediction without customer input details
    logger.info("Prediction generated successfully")

    return {
        "predicted_delivery_days": round(prediction, 2)
    }

"""Day 46: Production monitoring for the customer churn API.
Run with: python -m uvicorn day46:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "day43_customer_churn_model.joblib"
LOG_FILE = BASE_DIR / "day46_api.log"

logger = logging.getLogger("customer_churn_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
try:
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except OSError:
    logger.warning("Could not create log file; using console logging.")
logger.propagate = False

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Day 46: logging, validation, error handling, and monitoring.",
    version="2.0.0",
)
model = None
feature_columns = []
stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "churn_predictions": 0,
    "no_churn_predictions": 0,
}


class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: str = Field(default="Male", min_length=1)
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="Yes", min_length=1)
    Dependents: str = Field(default="No", min_length=1)
    tenure: int = Field(default=12, ge=0, le=1000)
    PhoneService: str = Field(default="Yes", min_length=1)
    MultipleLines: str = Field(default="No", min_length=1)
    InternetService: str = Field(default="DSL", min_length=1)
    OnlineSecurity: str = Field(default="No", min_length=1)
    OnlineBackup: str = Field(default="No", min_length=1)
    DeviceProtection: str = Field(default="No", min_length=1)
    TechSupport: str = Field(default="No", min_length=1)
    StreamingTV: str = Field(default="No", min_length=1)
    StreamingMovies: str = Field(default="No", min_length=1)
    Contract: str = Field(default="Month-to-month", min_length=1)
    PaperlessBilling: str = Field(default="Yes", min_length=1)
    PaymentMethod: str = Field(default="Electronic check", min_length=1)
    MonthlyCharges: float = Field(default=70.0, ge=0, allow_inf_nan=False)
    TotalCharges: float = Field(default=840.0, ge=0, allow_inf_nan=False)


@app.on_event("startup")
def load_model():
    global model, feature_columns
    if not MODEL_FILE.exists():
        logger.error("Model file missing: %s", MODEL_FILE.name)
        logger.error("Put the Day 43 .joblib model beside day46.py.")
        return
    try:
        model = joblib.load(MODEL_FILE)
        feature_columns = list(getattr(model, "feature_names_in_", []))
        logger.info("Model loaded successfully: %s", MODEL_FILE.name)
    except Exception:
        model = None
        logger.exception("Could not load saved model.")



@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    stats["total_requests"] += 1
    logger.info(
        "Request started id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"

        logger.info(
            "Request finished id=%s status=%s duration_ms=%.2f",
            request_id,
            response.status_code,
            elapsed,
        )
        return response

    except Exception:
        logger.exception("Request crashed id=%s", request_id)
        raise
@app.get("/")
async def home():
    return {
        "message": "Customer Churn Prediction API",
        "day": "46/60",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.get("/monitoring")
async def monitoring():
    return {
        "total_requests": stats["total_requests"],
        "status": "running"
    }



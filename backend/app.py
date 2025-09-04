# backend/app.py
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from . import schemas
from .recommender import Recommender
from .sentiment import SentimentModel
from .train_cf import build_and_save_cf
from . import config

# -----------------------
# Logging configuration
# -----------------------
os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(config.LOG_DIR, "app.log")

logger = logging.getLogger("ActiReco")
logger.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)

# Console handler (write to stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_formatter)

# File handler (utf-8 to allow emojis)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
file_handler.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)

# attach handlers if not already attached (prevents duplicates on reload)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="ActiReco API", version="0.2.7")

# Load heavy models once (may take time)
recommender = Recommender()
sentiment_model = SentimentModel()

# -----------------------
# Metrics store (in-memory, simple)
# -----------------------
metrics_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_latency": 0.0, "avg_latency": 0.0})

# -----------------------
# Middleware for latency logging
# -----------------------
@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {e}")
        # Return generic error (don't reveal internals)
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal server error"})

    process_time = (time.time() - start_time) * 1000.0  # ms
    endpoint = request.url.path

    # update metrics
    m = metrics_data[endpoint]
    m["count"] += 1
    m["total_latency"] += process_time
    m["avg_latency"] = m["total_latency"] / m["count"]

    logger.info(f"{request.method} {endpoint} → status={response.status_code} latency={process_time:.2f}ms")
    return response

# -----------------------
# Exception handlers
# -----------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException on {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(status_code=422, content={"status": "error", "detail": errors})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled global error: {exc}")
    return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal server error"})

# -----------------------
# Lifecycle hooks
# -----------------------
@app.on_event("startup")
def startup_event():
    logger.info("ActiReco API starting up")
    if not config.ADMIN_API_KEY:
        logger.warning("ADMIN_API_KEY not set. Admin endpoints will be disabled until ADMIN_API_KEY is configured.")
    logger.info(f"Running in debug={config.DEBUG}")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("ActiReco API shutting down")

# -----------------------
# Admin API key dependency
# -----------------------
def verify_admin_key(x_api_key: str = Header(...)):
    """
    Validate incoming X-API-KEY header against ADMIN_API_KEY from config.
    If ADMIN_API_KEY is not set, refuse admin calls to be explicit.
    """
    if not config.ADMIN_API_KEY:
        # explicit refusal: admin endpoints not enabled
        logger.warning("Received admin call but ADMIN_API_KEY is not configured.")
        raise HTTPException(status_code=503, detail="Admin endpoints not configured on this server.")
    if x_api_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return True

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    """Return average latency and call counts for each endpoint."""
    return {"endpoints": metrics_data}

# --- Sentiment ---
@app.post("/sentiment", response_model=schemas.SentimentResponse)
def detect_sentiment(req: schemas.SentimentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    mood = sentiment_model.analyze(req.text)
    return {"text": req.text, "mood": mood}

# --- Recommend (no sentiment) ---
@app.post("/recommend", response_model=schemas.RecommendationResponse)
def recommend(req: schemas.RecommendRequest):
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    items = recommender.recommend(
        user_id=req.user_id,
        top_k=req.top_k,
        mood=None,
        filter_seen=not req.include_seen,
        city=req.city,
        tags=req.tags,
        alpha_override=req.alpha,
        interests_override=req.interests_override,
    )
    if not items:
        # return empty list rather than 404 is another valid choice;
        # you previously raised 404 — keep that behaviour but it's also OK to return empty list
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {req.user_id}")
    return {"user_id": req.user_id, "recommendations": items}

# --- Recommend (with sentiment) ---
@app.post("/recommend_with_mood", response_model=schemas.RecommendationResponse)
def recommend_with_mood(req: schemas.RecommendRequest):
    mood = sentiment_model.analyze(req.mood_text) if req.mood_text else None
    items = recommender.recommend(
        user_id=req.user_id,
        top_k=req.top_k,
        mood=mood,
        filter_seen=not req.include_seen,
        city=req.city,
        tags=req.tags,
        alpha_override=req.alpha,
        interests_override=req.interests_override,
    )
    if not items:
        raise HTTPException(status_code=404, detail=f"No mood-based recommendations found for user {req.user_id}")
    return {"user_id": req.user_id, "mood": mood, "recommendations": items}

# --- Log interaction ---
@app.post("/log_interaction", response_model=schemas.LogInteractionResponse)
def log_interaction(req: schemas.LogInteractionRequest):
    import pandas as pd
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base, "data", "interactions.csv")

    if req.event == "rate" and req.rating is None:
        raise HTTPException(status_code=400, detail="Rating must be provided when event is 'rate'")

    new_row = {"user_id": req.user_id, "activity_id": req.activity_id, "event": req.event, "rating": req.rating}
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        df.to_csv(data_path, index=False)
        logger.info(f"Interaction logged: {new_row}")
        return {"status": "ok", "detail": "Logged"}
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to log interaction")

# --- Admin retrain CF ---
@app.post("/admin/retrain_cf", response_model=schemas.RetrainResponse, dependencies=[Depends(verify_admin_key)])
def retrain_cf(req: schemas.RetrainCFRequest):
    try:
        build_and_save_cf(n_factors=req.n_factors)
        logger.info(f"CF retrained with n_factors={req.n_factors}")
        return {"status": "ok", "detail": f"CF retrained with n_factors={req.n_factors}"}
    except Exception as e:
        logger.error(f"CF retraining failed: {e}")
        raise HTTPException(status_code=500, detail="CF retraining failed")
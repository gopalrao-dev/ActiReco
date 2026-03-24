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
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from .recommender import Recommender
from .sentiment import SentimentModel
from .train_cf import build_and_save_cf
from . import config
from .db import get_connection
from .analytics import get_popular_activities, get_user_analytics


# -----------------------
# Logging
# -----------------------
os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(config.LOG_DIR, "app.log")

logger = logging.getLogger("ActiReco")
logger.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# -----------------------
# App
# -----------------------
app = FastAPI(title="ActiReco API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.recommender = None
app.state.sentiment_model = None


# -----------------------
# Metrics
# -----------------------
metrics_data: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"count": 0, "total_latency": 0.0, "avg_latency": 0.0}
)


# -----------------------
# Middleware
# -----------------------
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(status_code=500, content={"status": "error"})

    latency = (time.time() - start) * 1000
    path = request.url.path

    m = metrics_data[path]
    m["count"] += 1
    m["total_latency"] += latency
    m["avg_latency"] = m["total_latency"] / m["count"]

    logger.info(f"{request.method} {path} {latency:.2f}ms")

    return response


# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def startup():
    logger.info("Starting ActiReco...")

    app.state.recommender = Recommender()
    app.state.sentiment_model = SentimentModel()


# -----------------------
# Exceptions
# -----------------------
@app.exception_handler(HTTPException)
async def http_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": exc.errors()})


@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.exception(str(exc))
    return JSONResponse(status_code=500, content={"error": "internal"})


# -----------------------
# Auth
# -----------------------
def verify_admin_key(x_api_key: str = Header(...)):
    if not config.ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin not configured")

    if x_api_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid key")

    return True


# -----------------------
# Core APIs
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {"endpoints": metrics_data}


@app.post("/sentiment", response_model=schemas.SentimentResponse)
def sentiment(req: schemas.SentimentRequest):
    mood = app.state.sentiment_model.analyze(req.text)
    return {"text": req.text, "mood": mood}


@app.post("/recommend", response_model=schemas.RecommendationResponse)
def recommend(req: schemas.RecommendRequest):
    items = app.state.recommender.recommend(**req.dict())
    return {"user_id": req.user_id, "recommendations": items}


@app.post("/recommend_with_mood", response_model=schemas.RecommendationResponse)
def recommend_mood(req: schemas.RecommendRequest):
    mood = app.state.sentiment_model.analyze(req.mood_text) if req.mood_text else None
    items = app.state.recommender.recommend(**req.dict(), mood=mood)
    return {"user_id": req.user_id, "mood": mood, "recommendations": items}


@app.post("/log_interaction", response_model=schemas.LogInteractionResponse)
def log_interaction(req: schemas.LogInteractionRequest):

    if req.event == "rate" and req.rating is None:
        raise HTTPException(status_code=400, detail="Rating required")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interactions (user_id, activity_id, event, rating)
        VALUES (?, ?, ?, ?)
    """, (req.user_id, req.activity_id, req.event, req.rating))

    conn.commit()
    conn.close()

    return {"status": "ok"}


# -----------------------
# 🔥 ANALYTICS APIs
# -----------------------
@app.get("/analytics/popular")
def popular():
    return {"data": get_popular_activities()}


@app.get("/analytics/user/{user_id}")
def user_analytics(user_id: str):
    return get_user_analytics(user_id)


# -----------------------
# Admin
# -----------------------
@app.post("/admin/retrain_cf", dependencies=[Depends(verify_admin_key)])
def retrain(req: schemas.RetrainCFRequest):
    build_and_save_cf(n_factors=req.n_factors)
    app.state.recommender = Recommender()
    return {"status": "ok"}
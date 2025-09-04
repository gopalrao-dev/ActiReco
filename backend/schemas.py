# backend/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, constr, conlist, confloat

# --- Sentiment ---
class SentimentRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=2000)

class SentimentResponse(BaseModel):
    text: str
    mood: str

# --- Recommend ---
class RecommendRequest(BaseModel):
    user_id: constr(strip_whitespace=True, min_length=1, max_length=64)
    top_k: int = Field(5, ge=1, le=50)
    mood_text: Optional[constr(strip_whitespace=True, max_length=2000)] = None

    city: Optional[constr(strip_whitespace=True, max_length=100)] = None
    tags: Optional[
        conlist(constr(strip_whitespace=True, min_length=1, max_length=50), min_length=0, max_length=10)
    ] = None
    include_seen: bool = False
    alpha: Optional[confloat(ge=0.0, le=1.0)] = None
    interests_override: Optional[constr(strip_whitespace=True, max_length=500)] = None

class RecommendationItem(BaseModel):
    activity_id: str
    title: Optional[str]
    tags: Optional[str]
    city: Optional[str]
    score: float
    content_score: float
    cf_score: Optional[float]

class RecommendationResponse(BaseModel):
    user_id: str
    mood: Optional[str] = None
    recommendations: List[RecommendationItem]

# --- Interaction logging ---
class LogInteractionRequest(BaseModel):
    user_id: str
    activity_id: str
    event: Literal["view", "click", "like", "rate"] = "view"
    rating: Optional[int] = Field(None, ge=1, le=5)

class LogInteractionResponse(BaseModel):
    status: str
    detail: Optional[str] = None

# --- Admin retrain ---
class RetrainCFRequest(BaseModel):
    n_factors: int = Field(50, ge=2, le=512)

class RetrainResponse(BaseModel):
    status: str
    detail: Optional[str] = None
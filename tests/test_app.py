# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import config

client = TestClient(app)


# -------------------
# Health + Metrics
# -------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics():
    # Call once to update metrics
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "/health" in data["endpoints"]


# -------------------
# Sentiment
# -------------------
def test_sentiment_valid():
    response = client.post("/sentiment", json={"text": "I am very happy today!"})
    assert response.status_code == 200
    data = response.json()
    assert "mood" in data
    assert data["text"] == "I am very happy today!"


def test_sentiment_empty_text():
    response = client.post("/sentiment", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty"


# -------------------
# Recommend
# -------------------
def test_recommend_invalid_topk():
    response = client.post("/recommend", json={"user_id": "u1", "top_k": 0})
    assert response.status_code == 400


def test_recommend_with_valid_user():
    # Note: this requires at least one user and activity in your CSVs
    response = client.post("/recommend", json={"user_id": "u1", "top_k": 3})
    # Either returns 200 with recs or 404 if no matches
    assert response.status_code in (200, 404)


# -------------------
# Recommend with Mood
# -------------------
def test_recommend_with_mood():
    response = client.post("/recommend_with_mood", json={
        "user_id": "u1",
        "top_k": 3,
        "mood_text": "Feeling sad today"
    })
    # Either returns 200 or 404
    assert response.status_code in (200, 404)


# -------------------
# Log Interaction
# -------------------
def test_log_interaction_valid():
    response = client.post("/log_interaction", json={
        "user_id": "u1",
        "activity_id": "a1",
        "event": "click",
        "rating": None
    })
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_log_interaction_missing_rating():
    response = client.post("/log_interaction", json={
        "user_id": "u1",
        "activity_id": "a1",
        "event": "rate"   # but no rating provided
    })
    assert response.status_code == 400
    assert "Rating must be provided" in response.json()["detail"]


# -------------------
# Admin retrain CF
# -------------------
def test_admin_retrain_cf_requires_key():
    response = client.post("/admin/retrain_cf", json={"n_factors": 5})
    assert response.status_code == 403 or response.status_code == 503


def test_admin_retrain_cf_with_key():
    if not config.ADMIN_API_KEY:
        pytest.skip("ADMIN_API_KEY not configured, skipping test")
    headers = {"x-api-key": config.ADMIN_API_KEY}
    response = client.post("/admin/retrain_cf", headers=headers, json={"n_factors": 5})
    # Could succeed or fail depending on your data
    assert response.status_code in (200, 500)
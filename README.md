# ActiReco — Activity Recommender (FastAPI)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal)  
![Tests](https://img.shields.io/badge/tests-passing-green)  
![License](https://img.shields.io/badge/license-MIT-lightgrey)  

**ActiReco** is a hybrid recommender system (content-based + collaborative filtering) with optional sentiment-aware recommendations.  
It provides a FastAPI backend, model training scripts, CF trainer (SVD), and a HuggingFace sentiment pipeline.

---

## 🚀 Quick Start

1. **Create virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
# for dev/test
pip install pytest httpx
# for HuggingFace sentiment model
pip install transformers torch
```

3. **Add `.env` file** in project root

```
ADMIN_API_KEY=supersecret123
HOST=127.0.0.1
PORT=8000
DEBUG=false
```

4. **Train CF model** (optional)

```bash
python backend/train_cf.py
```

5. **Run server**

```bash
python run.py
# or development
uvicorn backend.app:app --reload --port 8000
```

6. **Open docs** → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📂 Project Structure

```
.
├── run.py
├── .env
├── requirements.txt
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── recommender.py
│   ├── sentiment.py
│   ├── schemas.py
│   ├── train_cf.py
│   ├── train_recommender.py
│   └── models/
├── data/
│   ├── activities.csv
│   ├── interactions.csv
│   └── users.csv
├── logs/
│   └── app.log
├── tests/
│   └── test_app.py
└── README.md
```

---

## 🛠 API Reference

### Health
```
GET /health
```

### Sentiment Detection
```
POST /sentiment
{
  "text": "I feel amazing today!"
}
```

### Recommend
```
POST /recommend
{
  "user_id": "u1",
  "top_k": 3
}
```

### Recommend with Mood
```
POST /recommend_with_mood
{
  "user_id": "u1",
  "top_k": 3,
  "mood_text": "I feel excited"
}
```

### Admin Retrain CF
```
POST /admin/retrain_cf
Headers: X-API-KEY: supersecret123
{
  "n_factors": 50
}
```

---

## 🧪 Tests

Run unit tests:

```bash
pytest -q
```

---

## 📌 Notes

- HuggingFace model download on first run (~400MB)  
- Admin API requires `ADMIN_API_KEY` in `.env`  
- Persistence is CSV-based (upgradeable to DB later)  
- Logs stored in `logs/app.log`  

---

## 🔮 Future Work

- Switch persistence from **CSV → SQLite/Postgres**  
- Add **frontend integration** for activity browsing  
- Dockerize for easy deployment  
- CI/CD pipeline with GitHub Actions  
- Add more advanced recommenders (Neural CF, Transformers)  

---

## Summary

- FastAPI backend  
- Hybrid recommender (Content + CF)  
- Sentiment-aware recommendations  
- Admin retraining with API key  
- Metrics + latency logging  
- Pytest test suite  
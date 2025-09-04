# MuxConnect – Social Activity Recommender (Sentiment-Aware)
# ActiReco — Activity Recommender (FastAPI)

**ActiReco** is a small hybrid recommender (content-based + collaborative filtering) with optional sentiment-aware recommendations.
This repo contains a FastAPI backend, model training scripts, a simple CF trainer (SVD), and a HuggingFace sentiment pipeline.

---

## TL;DR Quick Start

1. Create a Python virtual environment and activate it.

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   # extra dev/test packages:
   pip install pytest httpx
   # if you want the HuggingFace sentiment model (recommended):
   pip install transformers torch
   ```
3. Add a `.env` file to the project root (example below).
4. Train the CF (optional; run if you want CF enabled):

   ```bash
   python backend/train_cf.py
   ```
5. Start the server:

   ```bash
   python run.py
   # or for development:
   uvicorn backend.app:app --reload --port 8000
   ```
6. Open docs: `http://127.0.0.1:8000/docs`

---

## Environment (`.env`)

Create `.env` in the **project root** (the folder that contains `run.py` and the `backend` directory).

Example `.env`:

```
ADMIN_API_KEY=supersecret123
HOST=127.0.0.1
PORT=8000
DEBUG=false
```

> You already have a `.gitignore` configured to ignore `.env` — keep secrets out of git.

---

## Install / Requirements

`requirements.txt` should contain the base runtime packages used by the project. Example minimum set (make sure your file includes these or run the install commands above):

```
fastapi
uvicorn
pandas
scikit-learn
joblib
numpy
scipy
transformers
torch
```

Dev/test extras:

```
pytest
httpx
```

Install all:

```bash
pip install -r requirements.txt
pip install pytest httpx
pip install transformers torch   # optional but needed for HF sentiment
```

---

## Build / Train Models

* Content-based artifacts: `backend/train_recommender.py` (if present) builds TF-IDF/vectorizer and saves into `backend/models/`.
* Collaborative filtering: run:

  ```bash
  python backend/train_cf.py
  ```

  This saves `cf_user_map.joblib`, `cf_item_map.joblib`, `cf_user_factors.npy`, `cf_item_factors.npy` under `backend/models/`.

If you don't train CF, the recommender will fall back to content-only (CF artifacts optional).

---

## Run server

Recommended (uses `backend/config.py`):

```bash
python run.py
```

Direct uvicorn:

```bash
uvicorn backend.app:app --reload --port 8000
```

Open API docs: `http://127.0.0.1:8000/docs`
Health: `http://127.0.0.1:8000/health`

---

## API Reference (key endpoints)

All request/response models are validated with Pydantic (see `backend/schemas.py`).

### Detect sentiment

`POST /sentiment`

```json
POST /sentiment
Content-Type: application/json
{
  "text": "I’m feeling really excited about sports this weekend!"
}
```

Response:

```json
{ "text": "...", "mood": "positive" }  // or "negative"/"neutral"
```

### Recommend (no mood)

`POST /recommend`

```json
{
  "user_id": "u1",
  "top_k": 3
}
```

### Recommend (with mood\_text)

`POST /recommend_with_mood`

```json
{
  "user_id": "u1",
  "top_k": 3,
  "mood_text": "I am feeling excited!"
}
```

`mood_text` will be analyzed by the sentiment model and used to boost content-based matches.

### Admin: Retrain CF

`POST /admin/retrain_cf` (requires `X-API-KEY` header)

```bash
curl -X POST "http://127.0.0.1:8000/admin/retrain_cf" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: supersecret123" \
  -d '{"n_factors": 50}'
```

---

## Example curl calls

Recommend:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","top_k":3}'
```

Recommend with mood:

```bash
curl -X POST "http://127.0.0.1:8000/recommend_with_mood" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","top_k":3,"mood_text":"I feel energetic"}'
```

Retrain CF (admin):

```bash
curl -X POST "http://127.0.0.1:8000/admin/retrain_cf" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: supersecret123" \
  -d '{"n_factors":50}'
```

---

## Tests (pytest)

You already created `tests/test_app.py`. To run tests:

```bash
# inside activated virtualenv and with .env filled
pytest -q
```

If tests need a running app (they should use TestClient), ensure dependencies are installed (`pytest`, `httpx`) and that your `backend.app` imports do not block (models load quickly or mocks are used).

If a test fails due to HuggingFace download latency, run the app once to let the HF model cache.

---

## Project structure (relevant files)

```
.
├── run.py
├── .env
├── requirements.txt
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── recommender.py
│   ├── train_cf.py
│   ├── train_recommender.py
│   ├── sentiment.py
│   ├── models/             # saved artifacts
│   └── data/
│       ├── activities.csv
│       └── interactions.csv
├── notebooks/
├── tests/
│   └── test_app.py
└── README.md
```

---

## Troubleshooting / Notes

* **HuggingFace model download**: The first time `SentimentModel` initializes it will download the model (\~hundreds of MB). Be patient; the server logs show progress.
* **Windows console emoji encoding**: You already handled logging encoding; keep `file_handler` with `encoding="utf-8"`.
* **CF returns zeros**: If your dataset is tiny, precision\@K will be zero. Use more interactions for meaningful CF.
* **Admin key**: If `.env` missing `ADMIN_API_KEY`, admin endpoints will be disabled / throw errors. Keep `.env` in project root.
* **Switching CSV → DB**: Optional later; current tests and run use CSVs in `backend/data/`.
* **Git**: add `.env` to `.gitignore` (already done).

---

## (summary)

* FastAPI backend with content + CF hybrid recommender.
* CF training script `backend/train_cf.py` using truncated SVD.
* Recommender class merges normalized content & CF scores; supports filters (city, tags), mood boosting, and seen-item filtering.
* HuggingFace sentiment integration `backend/sentiment.py` (works but requires `transformers` + `torch`).
* Logging, latency middleware, metrics endpoint, admin retrain endpoint with API key dependency.
* A `run.py` wrapper using `backend/config.py`.
* A `tests/test_app.py` (pytest) basic test suite.

---

If you want, I’ll:

* Prepare a short `requirements-dev.txt` and `requirements.txt` you can drop into the repo, and
* Provide a minimal `GitHub Actions` workflow to run `pytest` on every push.

Want me to generate those now?

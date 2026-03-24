import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from .db import get_connection


class Recommender:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

        models_dir = os.path.join(os.path.dirname(__file__), "models")

        self.vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
        self.activity_tfidf = joblib.load(os.path.join(models_dir, "activity_tfidf.joblib"))
        self.activities_df = joblib.load(os.path.join(models_dir, "activities_df.joblib"))

        # CF (SVD)
        try:
            self.cf_user_map = joblib.load(os.path.join(models_dir, "cf_user_map.joblib"))
            self.cf_item_map = joblib.load(os.path.join(models_dir, "cf_item_map.joblib"))
            self.cf_user_factors = np.load(os.path.join(models_dir, "cf_user_factors.npy"))
            self.cf_item_factors = np.load(os.path.join(models_dir, "cf_item_factors.npy"))
            self.has_cf = True
        except:
            self.has_cf = False

        self.activity_ids_ordered = self.activities_df["activity_id"].astype(str).tolist()

    # -----------------------
    # USER INTEREST TEXT
    # -----------------------
    def _user_text(self, user_id):
        try:
            conn = get_connection()
            df = pd.read_sql(
                "SELECT interests FROM users WHERE user_id = ?",
                conn,
                params=(user_id,)
            )
            conn.close()

            if not df.empty:
                return df.iloc[0]["interests"].replace(";", " ").lower()
        except:
            pass

        return ""

    # -----------------------
    # CF SCORE
    # -----------------------
    def _cf_scores(self, user_id):
        if not self.has_cf or user_id not in self.cf_user_map:
            return None

        uidx = self.cf_user_map[user_id]
        user_vec = self.cf_user_factors[uidx]
        scores = np.dot(self.cf_item_factors, user_vec)

        return scores

    # -----------------------
    # NORMALIZATION
    # -----------------------
    def _normalize(self, arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return arr

        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if np.isclose(mx, mn):
            return np.zeros_like(arr)

        return (arr - mn) / (mx - mn)

    # -----------------------
    # POPULAR ITEMS
    # -----------------------
    def _get_popular(self):
        try:
            conn = get_connection()
            df = pd.read_sql("""
                SELECT activity_id, COUNT(*) as cnt
                FROM interactions
                GROUP BY activity_id
                ORDER BY cnt DESC
            """, conn)
            conn.close()

            return df["activity_id"].tolist()
        except:
            return []

    # -----------------------
    # MAIN RECOMMEND
    # -----------------------
    def recommend(
        self,
        user_id,
        top_k=5,
        mood=None,
        filter_seen=True,
        city=None,
        tags=None,
        alpha_override=None,
        interests_override=None
    ):

        # -----------------------
        # CONTENT SCORE
        # -----------------------
        user_text = interests_override or self._user_text(user_id)
        user_vec = normalize(self.vectorizer.transform([user_text]))
        content_scores = cosine_similarity(user_vec, self.activity_tfidf).flatten()

        # -----------------------
        # MOOD BOOST
        # -----------------------
        if mood:
            mood = mood.lower()
            boost = {
                "positive": ["hiking", "sports", "dance", "gaming"],
                "negative": ["yoga", "meditation", "relax"]
            }

            tags_series = self.activities_df["tags"].fillna("").str.lower()

            for i, t in enumerate(tags_series):
                if any(k in t for k in boost.get(mood, [])):
                    content_scores[i] += 0.15

        # -----------------------
        # CF SCORE
        # -----------------------
        cf_scores = self._cf_scores(user_id)
        if cf_scores is None:
            cf_scores = np.zeros_like(content_scores)

        # -----------------------
        # HYBRID SCORING
        # -----------------------
        content_norm = self._normalize(content_scores)
        cf_norm = self._normalize(cf_scores)

        # dynamic alpha (smart)
        if user_text == "":
            alpha = 1.0   # cold start → content only
        else:
            alpha = alpha_override if alpha_override else self.alpha

        final_scores = alpha * content_norm + (1 - alpha) * cf_norm

        idxs = np.arange(len(final_scores))

        # -----------------------
        # FILTERS
        # -----------------------
        if city:
            city_col = self.activities_df["city"].fillna("").str.lower()
            idxs = idxs[city_col.iloc[idxs] == city.lower()]

        if tags:
            tcol = self.activities_df["tags"].fillna("").str.lower()
            idxs = [i for i in idxs if any(tag in tcol.iloc[i] for tag in tags)]

        # -----------------------
        # SEEN FILTER
        # -----------------------
        seen = set()

        if filter_seen:
            try:
                conn = get_connection()
                df = pd.read_sql(
                    "SELECT activity_id FROM interactions WHERE user_id = ?",
                    conn,
                    params=(user_id,)
                )
                conn.close()

                seen = set(df["activity_id"].astype(str))
            except:
                pass

        # -----------------------
        # RANKING
        # -----------------------
        order = np.argsort(-final_scores[idxs])

        results = []

        for j in order:
            idx = idxs[j]
            aid = str(self.activity_ids_ordered[idx])

            if aid in seen:
                continue

            row = self.activities_df.iloc[idx]

            # 🧠 EXPLANATION (NEW)
            reason = "Recommended based on your interests"
            if mood:
                reason += f" and your mood ({mood})"

            results.append({
                "activity_id": aid,
                "title": row.get("title"),
                "tags": row.get("tags"),
                "city": row.get("city"),
                "score": float(final_scores[idx]),
                "content_score": float(content_scores[idx]),
                "cf_score": float(cf_scores[idx]),
                "reason": reason
            })

            if len(results) >= top_k:
                break

        # -----------------------
        # FALLBACK
        # -----------------------
        if len(results) == 0:
            popular = self._get_popular()

            for aid in popular:
                row = self.activities_df[
                    self.activities_df["activity_id"] == aid
                ]

                if row.empty:
                    continue

                row = row.iloc[0]

                results.append({
                    "activity_id": aid,
                    "title": row.get("title"),
                    "tags": row.get("tags"),
                    "city": row.get("city"),
                    "score": 0.0,
                    "content_score": 0.0,
                    "cf_score": 0.0,
                    "reason": "Popular among users"
                })

                if len(results) >= top_k:
                    break

        return results
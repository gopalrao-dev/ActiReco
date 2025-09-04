# backend/recommender.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class Recommender:
    def __init__(self, alpha: float = 0.6):
        """
        alpha: weight for content-based score (0..1). 
               final_score = alpha*content + (1-alpha)*cf
        """
        self.alpha = alpha
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        # --- Load content artifacts ---
        vec_path = os.path.join(models_dir, "vectorizer.joblib")
        tfidf_path = os.path.join(models_dir, "activity_tfidf.joblib")
        df_path = os.path.join(models_dir, "activities_df.joblib")

        if not (os.path.exists(vec_path) and os.path.exists(tfidf_path) and os.path.exists(df_path)):
            # Auto-train content model if missing
            try:
                from .train_recommender import build_and_save_models
                build_and_save_models()
            except Exception as e:
                raise RuntimeError("Content artifacts missing and auto-train failed: " + str(e))

        self.vectorizer = joblib.load(vec_path)
        self.activity_tfidf = joblib.load(tfidf_path)  # sparse matrix
        self.activities_df = joblib.load(df_path)      # DataFrame

        # --- Load CF artifacts (persistent) ---
        cf_user_map_path = os.path.join(models_dir, "cf_user_map.joblib")
        cf_item_map_path = os.path.join(models_dir, "cf_item_map.joblib")
        cf_user_factors_path = os.path.join(models_dir, "cf_user_factors.npy")
        cf_item_factors_path = os.path.join(models_dir, "cf_item_factors.npy")

        if os.path.exists(cf_user_map_path) and os.path.exists(cf_item_map_path) and \
           os.path.exists(cf_user_factors_path) and os.path.exists(cf_item_factors_path):
            self.cf_user_map = joblib.load(cf_user_map_path)
            self.cf_item_map = joblib.load(cf_item_map_path)
            self.cf_user_factors = np.load(cf_user_factors_path)
            self.cf_item_factors = np.load(cf_item_factors_path)
            self.has_cf = True
            print("✅ CF model loaded from models/")
        else:
            self.cf_user_map = None
            self.cf_item_map = None
            self.cf_user_factors = None
            self.cf_item_factors = None
            self.has_cf = False
            print("⚠️ No CF model found. Run train_cf.py to build one.")

        # Precompute activity IDs
        if "activity_id" in self.activities_df.columns:
            self.activity_ids_ordered = self.activities_df["activity_id"].astype(str).tolist()
        else:
            self.activity_ids_ordered = [str(i) for i in range(len(self.activities_df))]

        # Load interactions
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        interactions_path = os.path.join(base, "data", "interactions.csv")
        if os.path.exists(interactions_path):
            try:
                self.interactions_df = pd.read_csv(interactions_path)
            except:
                self.interactions_df = None
        else:
            self.interactions_df = None

    # -----------------------
    # Helpers
    # -----------------------
    def _user_text_from_userid(self, user_id: str):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        users_path = os.path.join(base, "data", "users.csv")
        if os.path.exists(users_path):
            users_df = pd.read_csv(users_path)
            row = users_df[users_df["user_id"] == user_id]
            if not row.empty and "interests" in users_df.columns:
                interests = str(row.iloc[0]["interests"])
                return interests.replace(";", " ").lower()
        return ""

    def _cf_score_array(self, user_id: str):
        """Return CF score array aligned with activities_df order (same length)."""
        if not self.has_cf:
            return None
        if user_id not in self.cf_user_map:
            return None

        uidx = self.cf_user_map[user_id]
        if uidx >= self.cf_user_factors.shape[0]:
            return None

        user_vec = self.cf_user_factors[uidx]                # (k,)
        scores = np.dot(self.cf_item_factors, user_vec)     # (n_items,)
        return scores

    def _normalize(self, arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return arr
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if np.isclose(mx, mn):
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # -----------------------
    # Recommend API
    # -----------------------
    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        mood: str = None,
        filter_seen: bool = True,
        city: str = None,
        tags: list = None,
        alpha_override: float = None,
        interests_override: str = None
    ):
        """
        Returns list of dicts with activity info and scores.
        Supports:
          - mood: optional (positive/negative/neutral) boosts certain tags
          - filter_seen: remove items already interacted with
          - city: filter by city (exact, case-insensitive)
          - tags: list[str], keep item if ANY tag matches
          - alpha_override: override blend factor
          - interests_override: cold-start user text
        """
        # --- 1) Content-based scores ---
        user_text = interests_override if (interests_override and interests_override.strip()) \
                    else self._user_text_from_userid(user_id)
        user_vec = self.vectorizer.transform([user_text])
        user_vec = normalize(user_vec)
        content_scores = cosine_similarity(user_vec, self.activity_tfidf).flatten()

        # mood boosting
        if mood:
            mood = mood.lower()
            boost_map = {
                "positive": ["hiking","sports","dance","football","gaming","running","cycling","active","adventure"],
                "negative": ["yoga","meditation","journaling","relax","calm","mindfulness","therapy","spa"],
                "neutral": []
            }
            keywords = boost_map.get(mood, [])
            tags_series = self.activities_df["tags"].fillna("").astype(str).str.lower()
            for i, t in enumerate(tags_series):
                if any(kw in t for kw in keywords):
                    content_scores[i] += 0.15

        # --- 2) CF scores ---
        cf_scores = self._cf_score_array(user_id)
        if cf_scores is None:
            cf_scores = np.zeros_like(content_scores)

        # --- 3) Normalize + combine ---
        content_norm = self._normalize(content_scores)
        cf_norm = self._normalize(cf_scores)
        alpha = float(alpha_override) if (alpha_override is not None) else float(self.alpha)
        final_scores = alpha * content_norm + (1.0 - alpha) * cf_norm

        # --- 4) Candidate filtering ---
        idxs = np.arange(len(final_scores))

        # City filter
        if city:
            city_l = city.strip().lower()
            city_col = self.activities_df["city"].fillna("").astype(str).str.lower()
            idxs = idxs[city_col.iloc[idxs].values == city_l]

        # Tags filter
        if tags and len(tags) > 0:
            tags_l = [t.strip().lower() for t in tags]
            tcol = self.activities_df["tags"].fillna("").astype(str).str.lower()
            mask = []
            for i in idxs:
                itags = tcol.iloc[i]
                mask.append(any(tag in itags for tag in tags_l))
            idxs = idxs[np.array(mask, dtype=bool)]

        # Filter seen
        seen_set = set()
        if filter_seen and (self.interactions_df is not None):
            rows = self.interactions_df[self.interactions_df["user_id"] == user_id]
            seen_set = set(rows["activity_id"].astype(str).tolist())

        # --- 5) Rank & build results ---
        order = np.argsort(-final_scores[idxs])
        results = []
        for j in order:
            idx = idxs[j]
            aid = str(self.activity_ids_ordered[idx])
            if aid in seen_set:
                continue
            row = self.activities_df.iloc[idx]
            results.append({
                "activity_id": aid,
                "title": row.get("title"),
                "tags": row.get("tags"),
                "city": row.get("city") if "city" in row.index else None,
                "score": float(final_scores[idx]),
                "content_score": float(content_scores[idx]),
                "cf_score": float(cf_scores[idx]) if cf_scores is not None else None
            })
            if len(results) >= top_k:
                break

        return results

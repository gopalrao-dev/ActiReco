# backend/train_cf.py
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib

def build_and_save_cf(n_factors: int = 50):
    """
    Build Collaborative Filtering (CF) model using TruncatedSVD
    and persist all artifacts under backend/models/.
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    interactions_path = os.path.join(base, "data", "interactions.csv")
    activities_path = os.path.join(base, "data", "activities.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    inter = pd.read_csv(interactions_path)
    activities = pd.read_csv(activities_path)

    # consistent item ordering based on activities.csv
    item_ids = activities["activity_id"].astype(str).tolist()
    item_to_idx = {aid: i for i, aid in enumerate(item_ids)}
    n_items = len(item_ids)

    # users from interactions.csv
    user_ids = inter["user_id"].astype(str).unique().tolist()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    # Build lists for sparse matrix
    rows, cols, data = [], [], []
    for _, r in inter.iterrows():
        uid = str(r["user_id"])
        aid = str(r["activity_id"])
        if uid not in user_to_idx or aid not in item_to_idx:
            continue
        uidx = user_to_idx[uid]
        iidx = item_to_idx[aid]
        rating = r.get("rating", None)
        if pd.isna(rating) or rating is None:
            rating = r.get("liked", 1)
        try:
            val = float(rating)
        except:
            val = 1.0
        rows.append(uidx)
        cols.append(iidx)
        data.append(val)

    if len(data) == 0:
        raise RuntimeError("No interaction data found matching activities/users. Check interactions.csv and activities.csv.")

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # determine n_components safely
    max_comp = min(n_factors, min(R.shape) - 1)
    if max_comp <= 0:
        max_comp = 2

    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    user_factors = svd.fit_transform(R)       # shape (n_users, k)
    item_factors = svd.components_.T          # shape (n_items, k)

    # Save artifacts persistently
    joblib.dump(user_to_idx, os.path.join(models_dir, "cf_user_map.joblib"))
    joblib.dump(item_to_idx, os.path.join(models_dir, "cf_item_map.joblib"))
    np.save(os.path.join(models_dir, "cf_user_factors.npy"), user_factors)
    np.save(os.path.join(models_dir, "cf_item_factors.npy"), item_factors)
    joblib.dump(svd, os.path.join(models_dir, "cf_svd.joblib"))

    print("✅ CF model saved to:", models_dir)
    print("Users:", user_factors.shape[0], "Items:", item_factors.shape[0], "Latent dim:", user_factors.shape[1])

if __name__ == "__main__":
    build_and_save_cf()
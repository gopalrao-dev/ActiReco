#backend/train_lightfm.py

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import joblib
import os
from .db import get_connection

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_lightfm():
    conn = get_connection()

    interactions = pd.read_sql("SELECT user_id, activity_id FROM interactions", conn)
    activities = pd.read_sql("SELECT activity_id, tags FROM activities", conn)

    conn.close()

    # -----------------------
    # Build dataset
    # -----------------------
    dataset = Dataset()
    dataset.fit(
        users=interactions["user_id"].unique(),
        items=activities["activity_id"].unique()
    )

    # -----------------------
    # Build interactions matrix
    # -----------------------
    interactions_matrix, _ = dataset.build_interactions(
        [(row["user_id"], row["activity_id"]) for _, row in interactions.iterrows()]
    )

    # -----------------------
    # Item features (tags)
    # -----------------------
    activities["tags_list"] = activities["tags"].fillna("").apply(lambda x: x.split(";"))

    dataset.fit_partial(
        items=activities["activity_id"],
        item_features=[tag for tags in activities["tags_list"] for tag in tags]
    )

    item_features = dataset.build_item_features(
        [(row["activity_id"], row["tags_list"]) for _, row in activities.iterrows()]
    )

    # -----------------------
    # Train model
    # -----------------------
    model = LightFM(loss="warp")

    model.fit(
        interactions_matrix,
        item_features=item_features,
        epochs=20,
        num_threads=2
    )

    # -----------------------
    # Save everything
    # -----------------------
    joblib.dump(model, os.path.join(MODELS_DIR, "lightfm_model.joblib"))
    joblib.dump(dataset, os.path.join(MODELS_DIR, "lightfm_dataset.joblib"))
    joblib.dump(item_features, os.path.join(MODELS_DIR, "lightfm_item_features.joblib"))

    print("✅ LightFM model trained and saved!")


if __name__ == "__main__":
    train_lightfm()
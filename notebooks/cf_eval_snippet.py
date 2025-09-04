import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from backend.recommender import Recommender
from backend.train_cf import build_and_save_cf

# --------------------------
# Load and prepare data
# --------------------------
inter = pd.read_csv("data/interactions.csv")
inter = inter.sort_values("timestamp")

# Train-test split: last action of each user = test
test_rows = inter.groupby("user_id").tail(1)
train = inter.drop(test_rows.index)

# --------------------------
# Metrics
# --------------------------
def precision_at_k(recommended_ids, true_id, k=5):
    return 1.0 if true_id in recommended_ids[:k] else 0.0

def evaluate(rec, test_rows, ks=[1, 3, 5]):
    scores = {k: [] for k in ks}
    y_true, y_pred = [], []

    for _, row in test_rows.iterrows():
        uid = row["user_id"]
        true_item = str(row["activity_id"])
        true_rating = float(row.get("rating", 1))

        try:
            recs = rec.recommend(uid, top_k=10, filter_seen=False)
            rec_ids = [r["activity_id"] for r in recs]

            # Precision@K
            for k in ks:
                scores[k].append(precision_at_k(rec_ids, true_item, k=k))

            # RMSE: get predicted CF score for true item
            score_dict = {r["activity_id"]: r["cf_score"] for r in recs}
            if true_item in score_dict and score_dict[true_item] is not None:
                y_true.append(true_rating)
                y_pred.append(score_dict[true_item])

        except Exception:
            continue

    precision = {k: np.mean(scores[k]) if len(scores[k]) > 0 else 0.0 for k in ks}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else None
    return precision, rmse

# --------------------------
# Hyperparameter search
# --------------------------
alphas = [0.2, 0.5, 0.8]        # content vs CF blend
factors = [20, 50, 100]         # latent dimensions for CF

results = []

for nf in factors:
    print(f"\n=== Training CF with n_factors={nf} ===")
    build_and_save_cf(n_factors=nf)   # retrain CF for each n_factors

    for a in alphas:
        rec = Recommender(alpha=a)
        precision, rmse = evaluate(rec, test_rows)

        results.append({
            "n_factors": nf,
            "alpha": a,
            **precision,
            "rmse": rmse
        })

        print(f"n_factors={nf}, alpha={a}")
        for k, v in precision.items():
            print(f"  Precision@{k}: {v:.4f} (n={len(test_rows)})")
        print(f"  RMSE: {rmse if rmse is not None else 'N/A'}")

# --------------------------
# Save results to CSV
# --------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("notebooks/hyperparam_results.csv", index=False)
print("\nSaved results to notebooks/hyperparam_results.csv")
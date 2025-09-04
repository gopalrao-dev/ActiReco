# backend/train_recommender.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib

def build_and_save_models():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base, "data", "activities.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # load activities
    df = pd.read_csv(data_path)
    # Some defensive cleaning in case tag separator is ';'
    df["tags"] = df["tags"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["tags"].str.replace(";", " ")).str.lower()

    # Build TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(df["text"].tolist())
    X = normalize(X)  # normalize rows (good practice for cosine similarity)

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.joblib"))
    joblib.dump(X, os.path.join(models_dir, "activity_tfidf.joblib"))
    joblib.dump(df, os.path.join(models_dir, "activities_df.joblib"))

    print("Saved models to:", models_dir)

if __name__ == "__main__":
    build_and_save_models()
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.post_detect.p3_pca import (engineer_features, compute_lexical_diversity, normalize_features, apply_pca)
from src.post_detect.p4_ml_agent import (train_isolation_forest, train_lof)

# Get the cleaned POST dataset path
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")
# Inittial declaration
training_df = None

def get_training_dataset() -> pd.DataFrame:
    global training_df
    if training_df is None:
        training_df = pd.read_csv(CLEAN_PATH)

    return training_df


def get_authenticate_score(val: float, ref_vals: np.ndarray) -> int:
    return int(np.clip(np.mean(ref_vals <= val) * 100, 1, 100))


POST_CSV_COLUMN_MAP = {
    "like_count": "likes", 
    "text": "content", 
    "caption": "content",
    "comment_count": "comments", 
    "share_count": "shares",
    "char_count": "character_count", 
    "hashtags": "hashtag_count",
    "mentions": "mention_count", 
    "has_url": "contains_url",
    "media": "has_media", 
    "hour": "hour_of_day", 
    "day": "day_of_week",
}

POST_COLUMN_USE = ["likes", "comments", "shares", "hour_of_day", "day_of_week",
                  "is_weekend", "has_media", "character_count", "hashtag_count",
                  "mention_count", "contains_url", "content"]


def predict_batch_posts(rows: list) -> list:
    df = get_training_dataset()

    def normalize_dataset_row(row):
        csv_output = {}
        for key, value in row.items():
            key = key.strip()
            csv_output[POST_CSV_COLUMN_MAP.get(key.lower(), key.lower())] = value

        content = str(csv_output.get("content", ""))
        day = int(csv_output.get("day_of_week", 0) or 0)

        return {
            "likes": int(csv_output.get("likes", 0) or 0),
            "comments": int(csv_output.get("comments", 0) or 0),
            "shares": int(csv_output.get("shares", 0) or 0),
            "hour_of_day": int(csv_output.get("hour_of_day", 12) or 12),
            "day_of_week": day,
            "is_weekend": 1 if day in (5, 6) else int(csv_output.get("is_weekend", 0) or 0),
            "has_media": int(csv_output.get("has_media", 0) or 0),
            "character_count": len(content) if content else int(csv_output.get("character_count", 0) or 0),
            "hashtag_count": int(csv_output.get("hashtag_count", 0) or 0),
            "mention_count": int(csv_output.get("mention_count", 0) or 0),
            "contains_url": int(csv_output.get("contains_url", 0) or 0),
            "content": content,
        }

    # real_mask = (df["is_fake"] == 0).values
    input_rows = [normalize_dataset_row(r) for r in rows]
    input_df = pd.DataFrame(input_rows)

    df_combine = pd.concat([df[POST_COLUMN_USE], input_df[POST_COLUMN_USE]], ignore_index=True)
    X_eng = engineer_features(df_combine)
    X_scaled = StandardScaler().fit_transform(X_eng)
    post_pca, X_pca = apply_pca(X_scaled)

    CONTAMINATION = 0.5
    X_ref = X_pca[:len(df)]
    # X_train = X_ref[real_mask]
    X_train = X_pca

    if_model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, max_features=1.0, random_state=42)
    if_model.fit(X_train)
    if_scores_all = if_model.decision_function(X_pca)
    train_if = if_model.decision_function(X_train)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=CONTAMINATION)
    lof_model.fit(X_train)
    train_lof_scores = lof_model.decision_function(X_train)
    lof_input_scores = lof_model.decision_function(X_pca[len(df):])

    results = []
    for i, row in enumerate(rows):
        idx = len(df) + i
        if_s = get_authenticate_score(if_scores_all[idx], train_if)
        lof_s = get_authenticate_score(lof_input_scores[i], train_lof_scores)
        ensemble = int(round((if_s + lof_s) /2))
        content = input_rows[i]["content"]
        results.append({
            "if_score": if_s,
            "lof_score": lof_s,
            "ensemble_score": ensemble,
            "verdict": "Authentic" if ensemble >= 50 else "Suspicious",
            "lexical_diversity": round(compute_lexical_diversity(content), 4),
            "row_data": row,
        })

    return results


def predict_single_post(post_data: dict) -> dict:
    df = get_training_dataset()

    content = post_data.get("content", "")
    char_count = len(content) if content else int(post_data.get("character_count", 0))
    day_of_week = int(post_data.get("day_of_week", 0))

    single_row = {
        "likes": int(post_data.get("likes", 0)),
        "comments": int(post_data.get("comments", 0)),
        "shares": int(post_data.get("shares", 0)),
        "hour_of_day": int(post_data.get("hour_of_day", 12)),
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week in (5, 6) else int(post_data.get("is_weekend", 0)),
        "has_media": int(post_data.get("has_media", 0)),
        "character_count": char_count,
        "hashtag_count": int(post_data.get("hashtag_count", 0)),
        "mention_count": int(post_data.get("mention_count", 0)),
        "contains_url": int(post_data.get("contains_url", 0)),
        "content": content,
    }

    # real_mask = (df["is_fake"] == 0).values
    single_df = pd.DataFrame([single_row])
    df_combine = pd.concat([df[POST_COLUMN_USE], single_df[POST_COLUMN_USE]], ignore_index=True)

    X_eng = engineer_features(df_combine)
    X_scaled = StandardScaler().fit_transform(X_eng)
    post_pca, X_pca = apply_pca(X_scaled)

    CONTAMINATION = 0.5
    X_ref = X_pca[:len(df)]
    X_train = X_pca

    if_model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, max_features=1.0, random_state=42)
    if_model.fit(X_train)
    train_if = if_model.decision_function(X_train)
    if_score = get_authenticate_score(float(if_model.decision_function(X_pca[-1:])[0]), train_if)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=CONTAMINATION)
    lof_model.fit(X_train)
    train_lof = lof_model.decision_function(X_train)
    lof_score = get_authenticate_score(float(lof_model.decision_function(X_pca[-1:])[0]), train_lof)

    ensemble_score = int(round((if_score + lof_score) / 2))

    return {
        "if_score": if_score,
        "lof_score": lof_score,
        "ensemble_score": ensemble_score,
        "verdict": "Authentic" if ensemble_score >= 50 else "Suspicious",
        "lexical_diversity": round(compute_lexical_diversity(content), 4),
    }

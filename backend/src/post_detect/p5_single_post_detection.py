import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Get the cleaned POST dataset path
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")
# Inittial declaration
training_df = None

def get_training_dataset() -> pd.DataFrame:
    global training_df
    if training_df is None:
        training_df = pd.read_csv(CLEAN_PATH)

    return training_df


def compute_lexical_diversity(text: str) -> float:
    if not text or str(text).strip() == "":
        return 0.0
    words = str(text).lower().split()
    eps = 1e-6
    return len(set(words)) / (len(words) + eps)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    new_features = pd.DataFrame(index=df.index)

    encoded_categoricals = {"is_fake", "activity_id", "user_id", "post_country", "post_region", "post_city",
                            "device", "platform", "media_type", "content_type", "language"}
    
    numerical_columns = df.select_dtypes(include="number").columns

    for column in numerical_columns:
        if column not in encoded_categoricals:
            new_features[column] = df[column].astype(float)

    new_features["comments_per_like"] = df["comments"] / (df["likes"] + eps)
    new_features["shares_per_like"] = df["shares"] / (df["likes"] + eps)
    new_features["shares_per_comment"] = df["shares"] / (df["comments"] + eps)
    new_features["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    new_features["engagement_per_char"] = new_features["total_engagement"] / (df["character_count"] + eps)
    # new_features["hashtag_density"] = df["hashtag_count"] / (df["character_count"] + eps)
    # new_features["mention_density"] = df["mention_count"] / (df["character_count"] + eps)
    # new_features["url_per_char"] = df["contains_url"] / (df["character_count"] + eps)
    new_features["log_likes"] = np.log1p(df["likes"])
    new_features["log_comments"] = np.log1p(df["comments"])
    new_features["log_shares"] = np.log1p(df["shares"])
    new_features["log_character_count"] = np.log1p(df["character_count"])

    if "content" in df.columns:
        new_features["lexical_diversity"] = df["content"].apply(compute_lexical_diversity)

    return new_features


def get_authenticate_score(val: float, all_vals: np.ndarray) -> int:
    low, high = all_vals.min(), all_vals.max()
    if high == low:
        return 50
    return int(np.clip((val - low) / (high - low) * 100, 0, 100))


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

    input_rows = [normalize_dataset_row(r) for r in rows]
    input_df = pd.DataFrame(input_rows)

    combo = pd.concat([df[POST_COLUMN_USE], input_df[POST_COLUMN_USE]], ignore_index=True)
    X_eng = engineer_features(combo)
    X_scaled = StandardScaler().fit_transform(X_eng)

    cont = 0.35
    mask_normal = ~df["is_fake"].astype(bool)
    X_normal = X_scaled[:len(df)][mask_normal.values]

    if_model = IsolationForest(n_estimators=300, contamination=cont, max_features=1.0, random_state=42)
    if_model.fit(X_normal)
    if_scores_all = if_model.decision_function(X_scaled)
    train_if = if_scores_all[:len(df)]

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=cont)
    lof_model.fit(X_normal)
    train_lof_scores = lof_model.decision_function(X_normal)
    lof_input_scores = lof_model.decision_function(X_scaled[len(df):])

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
    single_df = pd.DataFrame([single_row])

    use_cols = ["likes", "comments", "shares", "hour_of_day", "day_of_week",
                "is_weekend", "has_media", "character_count", "hashtag_count",
                "mention_count", "contains_url", "content"]

    combo = pd.concat([df[use_cols], single_df[use_cols]], ignore_index=True)

    X_eng = engineer_features(combo)
    X_scaled = StandardScaler().fit_transform(X_eng)

    cont = 0.35
    mask_normal = ~df["is_fake"].astype(bool)
    X_normal = X_scaled[:len(df)][mask_normal.values]

    if_model = IsolationForest(n_estimators=300, contamination=cont, max_features=1.0, random_state=42)
    if_model.fit(X_normal)
    train_if = if_model.decision_function(X_normal)
    if_score = get_authenticate_score(float(if_model.decision_function(X_scaled[-1:])[0]), train_if)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=cont)
    lof_model.fit(X_normal)
    train_lof = lof_model.decision_function(X_normal)
    lof_score = get_authenticate_score(float(lof_model.decision_function(X_scaled[-1:])[0]), train_lof)

    ensemble_score = int(round(0.6 * if_score + 0.4 * lof_score))

    return {
        "if_score": if_score,
        "lof_score": lof_score,
        "ensemble_score": ensemble_score,
        "verdict": "Authentic" if ensemble_score >= 50 else "Suspicious",
        "lexical_diversity": round(compute_lexical_diversity(content), 4),
    }

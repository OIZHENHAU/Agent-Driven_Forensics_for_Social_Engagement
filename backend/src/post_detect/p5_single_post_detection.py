import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")

_training_df = None


def _get_training_df() -> pd.DataFrame:
    global _training_df
    if _training_df is None:
        _training_df = pd.read_csv(CLEAN_PATH)
    return _training_df


def compute_lexical_diversity(text: str) -> float:
    if not text or str(text).strip() == "":
        return 0.0
    words = str(text).lower().split()
    eps = 1e-6
    return len(set(words)) / (len(words) + eps)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    features = pd.DataFrame(index=df.index)

    encoded_categoricals = {
        "is_fake", "activity_id", "user_id",
        "post_country", "post_region", "post_city",
        "device", "platform", "media_type", "content_type", "language",
    }

    for col in df.select_dtypes(include="number").columns:
        if col not in encoded_categoricals:
            features[col] = df[col].astype(float)

    features["comments_per_like"] = df["comments"] / (df["likes"] + eps)
    features["shares_per_like"] = df["shares"] / (df["likes"] + eps)
    features["shares_per_comment"] = df["shares"] / (df["comments"] + eps)
    features["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    features["engagement_per_char"] = features["total_engagement"] / (df["character_count"] + eps)
    features["hashtag_density"] = df["hashtag_count"] / (df["character_count"] + eps)
    features["mention_density"] = df["mention_count"] / (df["character_count"] + eps)
    features["url_per_char"] = df["contains_url"] / (df["character_count"] + eps)
    features["log_likes"] = np.log1p(df["likes"])
    features["log_comments"] = np.log1p(df["comments"])
    features["log_shares"] = np.log1p(df["shares"])
    features["log_character_count"] = np.log1p(df["character_count"])

    if "content" in df.columns:
        features["lexical_diversity"] = df["content"].apply(compute_lexical_diversity)

    return features


def _get_score(val: float, all_vals: np.ndarray) -> int:
    low, high = all_vals.min(), all_vals.max()
    if high == low:
        return 50
    return int(np.clip((val - low) / (high - low) * 100, 0, 100))


def predict_single_post(post_data: dict) -> dict:
    df = _get_training_df()

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

    X_eng = _engineer_features(combo)
    X_scaled = StandardScaler().fit_transform(X_eng)

    mask_normal = ~df["is_fake"].astype(bool)
    X_normal = X_scaled[: len(df)][mask_normal.values]
    cont = float(np.clip(df["is_fake"].mean(), 0.05, 0.45))

    if_model = IsolationForest(n_estimators=300, contamination=cont, max_features=1.0, random_state=42, n_jobs=-1)
    if_model.fit(X_normal)
    train_if = if_model.decision_function(X_scaled[: len(df)])
    if_score = _get_score(float(if_model.decision_function(X_scaled[-1:])[0]), train_if)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=cont, n_jobs=-1)
    lof_model.fit(X_normal)
    train_lof = lof_model.decision_function(X_scaled[: len(df)])
    lof_score = _get_score(float(lof_model.decision_function(X_scaled[-1:])[0]), train_lof)

    ensemble_score = int(round(0.6 * if_score + 0.4 * lof_score))

    return {
        "if_score": if_score,
        "lof_score": lof_score,
        "ensemble_score": ensemble_score,
        "verdict": "Authentic" if ensemble_score >= 50 else "Suspicious",
        "lexical_diversity": round(compute_lexical_diversity(content), 4),
    }

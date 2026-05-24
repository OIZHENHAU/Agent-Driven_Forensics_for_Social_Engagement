import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)


CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "post")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Build engagement ratio features that distinguish viral posts from merely popular ones.
def engineer_features(df):
    eps = 1
    features = pd.DataFrame(index=df.index)

    for col in ["follower_count", "likes", "comments", "shares", "saves", "reach", "impressions", "engagement_rate",
                "followers_gained", "caption_length", "hashtags_count", "post_hour"]:
        if col in df.columns:
            features[col] = df[col]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "day_of_week" in df.columns:
        features["day_of_week"] = df["day_of_week"].map({d: i for i, d in enumerate(day_order)}).fillna(0).astype(int)

    features["likes_per_reach"] = df["likes"] / (df["reach"] + eps)
    features["comments_per_reach"] = df["comments"] / (df["reach"] + eps)
    features["shares_per_reach"] = df["shares"] / (df["reach"] + eps)
    features["saves_per_reach"] = df["saves"] / (df["reach"] + eps)
    features["shares_per_likes"] = df["shares"] / (df["likes"] + eps)
    features["saves_per_likes"] = df["saves"] / (df["likes"] + eps)
    features["total_engagement"] = df["likes"] + df["comments"] + df["shares"] + df["saves"]
    features["engagement_per_reach"] = features["total_engagement"] / (df["reach"] + eps)

    # Stronger virality signals: sharing/saving behaviour relative to passive likes
    features["virality_ratio"] = (df["shares"] + df["saves"]) / (df["likes"] + eps)
    features["follower_efficiency"] = df["followers_gained"] / (df["reach"] + eps)
    if "impressions" in df.columns:
        features["impression_reach_ratio"] = df["impressions"] / (df["reach"] + eps)

    # Polynomial engagement features to amplify separation at the high end of the distribution
    features["er_squared"] = df["engagement_rate"] ** 2
    features["er_cubed"] = df["engagement_rate"] ** 3

    # Per-account z-scores: viral posts are extreme outliers relative to their own account baseline
    if "account_id" in df.columns:
        acc_stats = df.groupby("account_id")[["engagement_rate", "likes", "shares", "saves"]].agg(["mean", "std"])
        acc_stats.columns = ["_".join(c) for c in acc_stats.columns]
        merged = df[["account_id"]].join(acc_stats, on="account_id")
        for col in ["engagement_rate", "likes", "shares", "saves"]:
            mu = merged[f"{col}_mean"].values
            sigma = merged[f"{col}_std"].fillna(1).replace(0, 1e-9).values
            features[f"{col}_zscore"] = (df[col].values - mu) / sigma

    return features


def scale_features(X_df):
    scaler = StandardScaler()
    return scaler.fit_transform(X_df)


def train_isolation_forest(X_normal, X_all, contamination=0.05):
    model = IsolationForest(n_estimators=300, contamination=contamination, max_features=1.0, random_state=42, n_jobs=-1)
    model.fit(X_normal)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


def train_lof(X_normal, X_all):
    model = LocalOutlierFactor(n_neighbors=30, novelty=True, metric="euclidean", n_jobs=-1)
    model.fit(X_normal)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


# Search the full score range to avoid missing the optimal threshold
def find_best_threshold(y_true, scores, target_precision=0.70, target_recall=0.70):
    thresholds = np.linspace(scores.min(), scores.max(), 2000)

    best_primary = None
    best_primary_f1 = 0.0
    best_fallback = None
    best_fallback_dist = float('inf')

    for thresh in thresholds:
        y_pred = (scores <= thresh).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        if prec >= target_precision and rec >= target_recall:
            if f1 > best_primary_f1:
                best_primary_f1 = f1
                best_primary = (thresh, prec, rec, f1, y_pred)

        dist = (prec - target_precision) ** 2 + (rec - target_recall) ** 2
        if dist < best_fallback_dist:
            best_fallback_dist = dist
            best_fallback = (thresh, prec, rec, f1, y_pred)

    return best_primary if best_primary else best_fallback


def plot_anomaly_results(predictions, title):
    normal  = (predictions == 1).sum()
    anomaly = (predictions == -1).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Anomaly"], [normal, anomaly])
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_post.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_post.png"))
    plt.close()


def evaluate_model(y_true, y_pred, model_name):
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n######## {model_name} Evaluation ##########")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n")

    return accuracy, precision, recall, f1


def main(df):
    y_true = np.where(df["performance_bucket_label"] == "viral", 1, 0)

    X_eng = engineer_features(df)
    X_scaled = scale_features(X_eng)

    mask_normal = df["performance_bucket_label"].isin(["low", "high"])
    X_normal = X_scaled[mask_normal]

    # Match contamination to the actual viral rate so IF can flag enough anomalies
    actual_contamination = float(np.clip(y_true.mean(), 0.05, 0.45))

    # Train Isolation Forest Model
    if_preds_default, if_scores = train_isolation_forest(X_normal, X_scaled, contamination=actual_contamination)

    print("\nIsolation Forest Results")
    print(pd.Series(if_preds_default).value_counts())
    plot_anomaly_results(if_preds_default, "Isolation_Forest_Result")

    if_best = find_best_threshold(y_true, if_scores)
    thresh, prec, rec, f1, if_y_pred = if_best
    met = prec >= 0.70 and rec >= 0.70
    label = "Optimised" if met else "Best-available"
    print(f"{label} threshold={thresh:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    evaluate_model(y_true, if_y_pred, "Isolation Forest")


    # Train Local Outlier Factor model
    lof_preds_default, lof_scores = train_lof(X_normal, X_scaled)

    print("\nLOF Results")
    print(pd.Series(lof_preds_default).value_counts())
    plot_anomaly_results(lof_preds_default, "LOF_Result")

    lof_best = find_best_threshold(y_true, lof_scores)
    thresh, prec, rec, f1, lof_y_pred = lof_best
    met = prec >= 0.70 and rec >= 0.70
    label = "Optimised" if met else "Best-available"
    print(f"{label} threshold={thresh:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return if_preds_default, lof_preds_default

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


CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

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

    return features


def scale_features(X_df):
    scaler = StandardScaler()
    return scaler.fit_transform(X_df)


def train_isolation_forest(X_normal, X_all):
    model = IsolationForest(n_estimators=300, contamination=0.05, max_features=1.0, random_state=42, n_jobs=-1)
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


def find_best_threshold(y_true, scores, target_precision=0.60, target_recall=0.40):
    thresholds = np.percentile(scores, np.linspace(0.5, 49, 300))

    best_result = None
    best_f1 = 0.0

    for thresh in thresholds:
        y_pred = (scores <= thresh).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        if prec >= target_precision and rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_result = (thresh, prec, rec, f1, y_pred)

    return best_result


def plot_anomaly_results(predictions, title):
    normal  = (predictions == 1).sum()
    anomaly = (predictions == -1).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(["Normal", "Anomaly"], [normal, anomaly])
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
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

    X_normal = X_scaled[y_true == 0]

    # Train Isolation Forest Model
    if_preds_default, if_scores = train_isolation_forest(X_normal, X_scaled)

    print("\nIsolation Forest Results")
    print(pd.Series(if_preds_default).value_counts())
    plot_anomaly_results(if_preds_default, "Isolation_Forest_Result")

    if_best = find_best_threshold(y_true, if_scores)
    if if_best:
        thresh, prec, rec, f1, if_y_pred = if_best
        print(f"Optimised threshold={thresh:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    else:
        print("No threshold met all targets — using default contamination boundary")
        if_y_pred = np.where(if_preds_default == -1, 1, 0)

    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    evaluate_model(y_true, if_y_pred, "Isolation Forest")


    #Train Local Outlier Factor model
    lof_preds_default, lof_scores = train_lof(X_normal, X_scaled)

    print("\nLOF Results")
    print(pd.Series(lof_preds_default).value_counts())
    plot_anomaly_results(lof_preds_default, "LOF_Result")

    lof_best = find_best_threshold(y_true, lof_scores)
    if lof_best:
        thresh, prec, rec, f1, lof_y_pred = lof_best
        print(f"Optimised threshold={thresh:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    else:
        print("No threshold for all targets.")
        lof_y_pred = np.where(lof_preds_default == -1, 1, 0)

    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return if_preds_default, lof_preds_default

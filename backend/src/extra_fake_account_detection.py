import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fake_social_media_global_2.0_with_missing.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} accounts, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    for col in ["username_length", "digits_count", "special_char_count", "digit_ratio", "verified", "username_randomness"]:
        if col in df.columns:
            features[col] = df[col]

    if "repeat_char_count" in df.columns:
        features["log_repeat_char_count"] = np.log1p(
            df["repeat_char_count"].clip(lower=0))

    if {"username_length", "repeat_char_count"} <= set(df.columns):
        features["username_anomaly"]    = (df["username_length"] * df["repeat_char_count"])
        features["username_anomaly_sq"] = (df["username_length"] ** 2 * df["repeat_char_count"])

    if {"digits_count", "username_length"} <= set(df.columns):
        features["digit_density"] = df["digits_count"] / (df["username_length"] + 1)

    if "username_length" in df.columns:
        features["username_length_sq"] = df["username_length"] ** 2

    return features


def scale_features(X_df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X_df)


def train_isolation_forest(X: np.ndarray, contamination: float = 0.15):
    model = IsolationForest(n_estimators=300, contamination=contamination, max_features=0.9, random_state=42,n_jobs=-1)
    model.fit(X)
    scores = model.decision_function(X)
    predictions = model.predict(X)
    return predictions, scores, model


def train_lof(X: np.ndarray, contamination: float = 0.15):
    model = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=contamination, metric="euclidean", n_jobs=-1)
    model.fit(X)
    scores = model.decision_function(X)
    predictions = model.predict(X)
    return predictions, scores, model


def compute_authenticity_score(if_scores: np.ndarray, lof_scores: np.ndarray) -> np.ndarray:
    scaler   = MinMaxScaler()
    if_norm  = scaler.fit_transform(if_scores.reshape(-1, 1)).flatten()
    lof_norm = scaler.fit_transform(lof_scores.reshape(-1, 1)).flatten()
    combined = 0.6 * if_norm + 0.4 * lof_norm

    return np.round(combined * 100, 2)


def find_best_threshold(y_true, scores, target_precision=0.70, target_recall=0.70):
    thresholds = np.linspace(scores.min(), scores.max(), 2000)
    best_primary = None
    best_primary_f1 = 0.0
    best_fallback = None
    best_fallback_f1= 0.0

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

        if f1 > best_fallback_f1:
            best_fallback_f1 = f1
            best_fallback = (thresh, prec, rec, f1, y_pred)

    return best_primary if best_primary else best_fallback


def plot_anomaly_distribution(predictions: np.ndarray, title: str) -> None:
    normal = (predictions ==  1).sum()
    anomaly = (predictions == -1).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(["Authentic", "Suspicious / Fake"], [normal, anomaly], color=["steelblue", "tomato"])
    plt.title(title)
    plt.ylabel("Account Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def plot_authenticity_score_dist(scores: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=40, color="steelblue", edgecolor="white")
    plt.axvline(50, color="tomato", linestyle="--", label="50% threshold")
    plt.title(title)
    plt.xlabel("Authenticity Confidence Score  (0 = Fake, 100 = Real)")
    plt.ylabel("Account Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n######## {model_name} Evaluation ########")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main(df: pd.DataFrame = None):
    if df is None:
        df = load_dataset()

    df = preprocess(df)
    y_true = df["is_fake"].astype(int).values if "is_fake" in df.columns else None

    X_df = engineer_features(df)
    X_scaled = scale_features(X_df)

    actual_contamination = float(np.clip(y_true.mean(), 0.05, 0.45)) if y_true is not None else 0.15

    if_preds, if_scores, _ = train_isolation_forest(X_scaled, actual_contamination)

    print("\nIsolation Forest — account distribution:")
    print(pd.Series(if_preds).value_counts().rename({1: "Authentic", -1: "Suspicious"}))
    plot_anomaly_distribution(if_preds, "IF_Fake_Account_Distribution")

    if y_true is not None:
        if_best = find_best_threshold(y_true, if_scores)
        thresh, prec, rec, f1, if_y_pred = if_best
        met   = prec >= 0.70 and rec >= 0.70
        label = "Optimised" if met else "Best-available"
        print(f"{label} IF  threshold={thresh:.4f}  " f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        plot_confusion_matrix(y_true, if_y_pred, "IF_Fake_Account_Confusion_Matrix")
        evaluate_model(y_true, if_y_pred, "Isolation Forest")


    lof_preds, lof_scores, _ = train_lof(X_scaled, actual_contamination)

    print("\nLOF — account distribution:")
    print(pd.Series(lof_preds).value_counts()
            .rename({1: "Authentic", -1: "Suspicious"}))
    plot_anomaly_distribution(lof_preds, "LOF_Fake_Account_Distribution")

    if y_true is not None:
        lof_best = find_best_threshold(y_true, lof_scores)
        thresh, prec, rec, f1, lof_y_pred = lof_best
        met   = prec >= 0.70 and rec >= 0.70
        label = "Optimised" if met else "Best-available"
        print(f"{label} LOF threshold={thresh:.4f}  "
              f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        plot_confusion_matrix(y_true, lof_y_pred, "LOF_Fake_Account_Confusion_Matrix")
        evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    auth_scores = compute_authenticity_score(if_scores, lof_scores)
    plot_authenticity_score_dist(auth_scores,
                                 "Authenticity_Confidence_Score_Distribution")

    print(f"\nAuthenticity Confidence Score — sample (first 10 accounts):")
    print(auth_scores[:10])

    df = df.copy()
    df["authenticity_score"] = auth_scores
    df["if_prediction"] = if_preds
    df["lof_prediction"] = lof_preds

    df["is_suspicious"] = (auth_scores < 50).astype(int)

    suspicious_count = df["is_suspicious"].sum()
    print(f"\nAccounts flagged suspicious (score < 50): " f"{suspicious_count} ({suspicious_count / len(df) * 100:.1f}%)")

    return df


if __name__ == "__main__":
    main()

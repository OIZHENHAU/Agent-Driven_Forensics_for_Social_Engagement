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


CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_account_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "account")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def engineer_features(df):
    eps = 1e-6
    features = pd.DataFrame(index=df.index)

    numerical_columns = (df.select_dtypes(include='number').columns)
    # Columns to remove
    drop_columns = ["fake"]

    # Keep only useful features
    important_features = [col for col in numerical_columns if col not in drop_columns]

    # Raw profile attributes
    for col in important_features:
        if col in df.columns:
            features[col] = df[col]

    # Enginering features
    features["follower_following_ratio"] = df["#followers"] / (df["#follows"] + eps)
    features["posts_per_follower"] = df["#posts"] / (df["#followers"] + eps)
    features["following_per_follower"] = df["#follows"] / (df["#followers"] + eps)
    features["posts_per_following"] = df["#posts"] / (df["#follows"] + eps)

    features["has_description"] = (df["description length"] > 0).astype(int)

    features["log_posts"] = np.log1p(df["#posts"])
    features["log_followers"] = np.log1p(df["#followers"])
    features["log_follows"] = np.log1p(df["#follows"])

    # features["username_nums_sq"] = df["nums/length username"] ** 2
    # features["fullname_nums_sq"] = df["nums/length fullname"] ** 2

    return features


def scale_features(X_df):
    scaler = StandardScaler()
    return scaler.fit_transform(X_df)


def train_isolation_forest(X_normal, X_all, contamination=0.45):
    model = IsolationForest(n_estimators=300, contamination=contamination, max_features=1.0, random_state=42, n_jobs=-1)
    model.fit(X_normal)
    scores      = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


def train_lof(X_normal, X_all):
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", n_jobs=-1, contamination=0.5)
    model.fit(X_normal)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


def find_best_threshold(y_true, scores, target_lo=0.70, target_hi=0.90):
    thresholds = np.linspace(scores.min(), scores.max(), 5000)
    target_mid = (target_lo + target_hi) / 2.0

    best_recall_near_mid = None
    best_recall_near_mid_dist = float("inf")
    best_fallback = None
    best_fallback_dist = float("inf")

    for thresh in thresholds:
        y_pred = (scores <= thresh).astype(int)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)


        if target_lo <= rec <= target_hi and prec >= target_lo and acc >= target_lo:
            d = abs(rec - target_mid)
            if d < best_recall_near_mid_dist:
                best_recall_near_mid_dist = d
                best_recall_near_mid = (thresh, prec, rec, acc, f1, y_pred)

        dist = (prec - target_mid) ** 2 + (rec - target_mid) ** 2 + (acc - target_mid) ** 2
        if dist < best_fallback_dist:
            best_fallback_dist = dist
            best_fallback = (thresh, prec, rec, acc, f1, y_pred)

    return best_recall_near_mid if best_recall_near_mid else best_fallback


def plot_anomaly_results(predictions, title):
    real = (predictions ==  1).sum()
    fake = (predictions == -1).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], [real, fake])
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n######## {model_name} Evaluation ##########")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print()

    return accuracy, precision, recall, f1


def main(df):
    y_true = df["fake"].values

    X_eng = engineer_features(df)
    X_scaled = scale_features(X_eng)

    mask_normal = (df["fake"] == 0).values
    X_normal = X_scaled[mask_normal]

    contamination = 0.5

    # Train Isolation Forest
    if_preds, _ = train_isolation_forest(X_normal, X_scaled, contamination=contamination)

    print("\nIsolation Forest raw predictions")
    print(pd.Series(if_preds).value_counts())
    plot_anomaly_results(if_preds, "Isolation_Forest_Result")

    if_y_pred = (if_preds == -1).astype(int)
    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    evaluate_model(y_true, if_y_pred, "Isolation Forest")

    # Train Local Outlier Factor
    lof_preds, _ = train_lof(X_normal, X_scaled)

    print("\nLOF raw predictions")
    print(pd.Series(lof_preds).value_counts())
    plot_anomaly_results(lof_preds, "LOF_Result")

    lof_y_pred = (lof_preds == -1).astype(int)
    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return if_preds, lof_preds


# training_model_account = main

# Train all the IsoaltionForest and LOF model on the cleaned dataset
def predict_single_account(raw_input: dict) -> dict:
    df = pd.read_csv(CLEAN_PATH)

    single_row = pd.DataFrame([{
        "profile pic": int(raw_input.get("profile_pic", 0)),
        "nums/length username": float(raw_input.get("nums_length_username", 0)),
        "fullname words": int(raw_input.get("fullname_words", 1)),
        "nums/length fullname": float(raw_input.get("nums_length_fullname", 0)),
        "name==username": int(raw_input.get("name_equals_username", 0)),
        "description length": int(raw_input.get("description_length", 0)),
        "external URL": int(raw_input.get("external_url", 0)),
        "private": int(raw_input.get("private", 0)),
        "#posts": int(raw_input.get("posts", 0)),
        "#followers": int(raw_input.get("followers", 0)),
        "#follows": int(raw_input.get("follows", 0)),
    }])

    df_features = df.drop(columns=["fake"])
    df_combined = pd.concat([df_features, single_row], ignore_index=True)

    X_eng = engineer_features(df_combined)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)

    CONTAMINATION = 0.5
    X_train = X_scaled[:len(df)]

    # Train Isolation Forest on input data
    if_model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, max_features=1.0, random_state=42, n_jobs=-1)
    if_model.fit(X_train)
    if_scores = if_model.decision_function(X_scaled)

    # Train LOF on the input data
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", n_jobs=-1, contamination=CONTAMINATION)
    lof_model.fit(X_train)
    lof_train_scores = lof_model.decision_function(X_train)
    lof_new_score = float(lof_model.decision_function(X_scaled[-1:])[0])

    def to_auth_score(score, ref_scores):
        s_min, s_max = ref_scores.min(), ref_scores.max()
        if s_max == s_min:
            return 50
        return int(np.clip((score - s_min) / (s_max - s_min) * 100, 1, 100))

    if_auth = to_auth_score(if_scores[-1], if_scores[:len(df)])
    lof_auth = to_auth_score(lof_new_score, lof_train_scores)
    ensemble = int(round(0.6 * if_auth + 0.4 * lof_auth))

    return {"if_score": if_auth, "lof_score": lof_auth, "ensemble_score": ensemble, "verdict": "Authentic" if ensemble >= 50 else "Suspicious"}


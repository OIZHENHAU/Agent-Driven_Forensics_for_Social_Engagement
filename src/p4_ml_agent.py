import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Get the cleaned dataset path
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
# Get the PCA dataset path
PCA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pca_data.csv")
# Get the importance features in the json file
JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pca_features.json")
# Declare prediction path
PRED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.csv")


# Colour to determine the real or fake account, grren represent real and red represent fake.
PALETTE = {0: "#2ecc71", 1: "#e74c3c"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

important_features = [
    "digit_ratio",
    "digits_count",
    "username_length",
    "special_char_count",
    "repeat_char_count",
    "follow_unfollow_rate",
    "posts_per_day",
    "follower_following_ratio",
    "suspicious_engagement_rate",
    "spam_comments_rate",
    "generic_comment_rate",
    "caption_similarity_score",
    "content_similarity_score",
    "username_randomness",
    "account_age_days",
    "followers",
    "following",
    "posts",
    "verified",
    "has_profile_pic",
    "suspicious_links_in_bio"
]

# Get the clean dataset
def load_dataset() -> pd.DataFrame:
    clean_dataset = pd.read_csv(CLEAN_PATH)
    return clean_dataset

# GEt the contamination fraction for fake account
def get_contamination():
    clean_dataset = load_dataset()
    print(clean_dataset["is_fake"].value_counts())
    fraction = len(clean_dataset[clean_dataset["is_fake"] == 1]) / float(len(clean_dataset))
    return fraction


# Retrieve the importance features 
def get_important_features() -> list:
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as important_features:
            f_json = json.load(important_features)
        all_features = [f for f in f_json["features"] if f in pd.read_csv(CLEAN_PATH).columns]
        return all_features
    return []


def load_and_prepare_data():
    # Get the cleaned dataset.
    df = load_dataset()
    
    # Get numeric columns, exclude column is fake, username and platform or any non-numerical columns
    exclude_columns = ['is_fake', 'username', 'platform']
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    features_needed = [col for col in numeric_columns if col not in exclude_columns]
    
    X = df[features_needed]
    y = df['is_fake']
    
    return X, y, features_needed


# Scale features and reduce skewness
def scale_features(X: pd.DataFrame, features: list):
    X = X.copy()
    # Check skewness
    skewness = X[features].skew()
    # Find highly skewed columns
    skewed_columns = skewness[abs(skewness) > 1].index

    print("Skewed columns:")
    print(skewed_columns)

    # Apply log transform
    for col in skewed_columns:
        X[col] = np.log1p(X[col].clip(lower=0))

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    return X_scaled


#Build the isolation forest machine learning
def run_isolation_forest(X: np.ndarray) -> tuple:
    contamination_fraction = get_contamination()

    forest = IsolationForest(n_estimators=200, contamination=contamination_fraction, random_state=42)
    raw_pred = forest.fit_predict(X)
    scores = forest.decision_function(X)
    forest_preds = np.where(raw_pred == -1, 1, 0)
    # Calculate the authentication score
    auth_score = ((scores - scores.min()) /(scores.max() - scores.min()) * 100).round(1)

    return forest_preds, auth_score, forest


#Build Local Outlier Factors (LOF)
def run_lof(X: np.ndarray) -> tuple:
    contamination_fraction = get_contamination()

    # Reduce dimensionality for LOF
    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_fraction)
    raw_pred = lof.fit_predict(X_pca)
    scores = -lof.negative_outlier_factor_
    lof_preds = np.where(raw_pred == -1, 1, 0)

    auth_score = (1 - ((scores - scores.min()) / (scores.max() - scores.min()))) * 100

    auth_score = auth_score.round(1)

    return lof_preds, auth_score, lof


#Calculate the evaluation score between y_true, & y_pred for each model 
def evaluate(y_true, y_pred, auth_score, model_name: str) -> dict:
    print(model_name)
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    prob_fake = 1 - auth_score / 100
    precision, recall, _ = precision_recall_curve(y_true, prob_fake)
    pr_auc = auc(recall, precision)

    roc_auc = roc_auc_score(y_true, prob_fake)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:\n{cm}")

    return {"model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc, "precision": precision, "recall": recall}


def plot_confusion(y_true, if_pred, lof_pred) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, preds, name in zip(axes, [if_pred, lof_pred], ["Isolation Forest", "LOF"]):
        cm  = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
        ax.set_title(f"{name} — Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ml_confusion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Saved → {path}")


def plot_pr_curves(results: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#3498db", "#e67e22"]

    for res, col in zip(results, colors):
        ax.plot(res["recall"], res["precision"], label=f"{res['model']}  (PR-AUC={res['pr_auc']:.3f})", color=col, lw=2)

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ml_pr_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Saved → {path}")


'''def plot_auth_score_dist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, name in zip(axes, ["if_auth_score", "lof_auth_score"], ["Isolation Forest", "LOF"]):
        for lab, label_name in [(0, "Real"), (1, "Fake")]:
            subset = df[df["is_fake"] == lab][col]
            ax.hist(subset, bins=40, alpha=0.6, color=PALETTE[lab], label=label_name, density=True)

        ax.set_title(f"{name} — Authenticity Score Distribution")
        ax.set_xlabel("Authenticity Score (0-100)")
        ax.set_ylabel("Density"); ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ml_auth_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Saved → {path}")
'''


def save_predictions(df_orig: pd.DataFrame,if_pred, if_score, lof_pred, lof_score) -> pd.DataFrame:
    out = df_orig[important_features].copy()
    out["if_pred"] = if_pred
    out["if_auth_score"]  = if_score
    out["lof_pred"] = lof_pred
    out["lof_auth_score"] = lof_score
    out["ensemble_pred"]  = ((if_pred == 1) & (lof_pred == 1)).astype(int)
    out["ensemble_score"] = ((if_score + lof_score) / 2).round(1)
    out.to_csv(PRED_PATH, index=False)
    print(f"[ML] Predictions saved → {PRED_PATH}")
    return out


def main():
    clean_dataset = load_dataset()
    X, y, feature_names = load_and_prepare_data()
    X_scaled = scale_features(X, important_features)
    y_true = clean_dataset["is_fake"].values

    forest_pred, forest_score, forest = run_isolation_forest(X_scaled)
    lof_pred, lof_score, lof = run_lof(X_scaled)

    forest_evaluation_result  = evaluate(y_true, forest_pred, forest_score, "Isolation Forest")
    lof_evaluation_result = evaluate(y_true, lof_pred, lof_score, "LOF")

    plot_confusion(y_true, forest_pred, lof_pred)
    plot_pr_curves([forest_evaluation_result, lof_evaluation_result])

    # pred_df = save_predictions(clean_dataset, forest_pred, forest_score, lof_pred, lof_score)
    # plot_auth_score_dist(pred_df)


if __name__ == "__main__":
    main()
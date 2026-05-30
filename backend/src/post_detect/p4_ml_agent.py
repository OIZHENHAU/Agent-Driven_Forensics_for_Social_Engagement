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

# Caluclate the lexical diversity score by using the Type-Token RAtio (unique words / total words)
def compute_lexical_diversity(text_series):
    eps = 1e-6
    def ttr(text):
        if pd.isna(text) or str(text).strip() == "":
            return 0.0
        words = str(text).lower().split()
        return len(set(words)) / (len(words) + eps)
    
    return text_series.apply(ttr)


def engineer_features(df):
    eps = 1e-6
    features = pd.DataFrame(index=df.index)

    # Exclude label-encoded categoricals
    encoded_categoricals = {
        "is_fake", "activity_id", "user_id",
        "post_country", "post_region", "post_city",
        "device", "platform", "media_type", "content_type", "language",
    }

    # Get all numerical columns
    numerical_columns = (df.select_dtypes(include='number').columns)

    # Raw numeric signals (skip encoded categoricals)
    for col in numerical_columns:
        if col in df.columns and col not in encoded_categoricals:
            features[col] = df[col].astype(float)

    # Engagement ratios
    features["comments_per_like"] = df["comments"] / (df["likes"] + eps)
    features["shares_per_like"] = df["shares"] / (df["likes"] + eps)
    features["shares_per_comment"]  = df["shares"] / (df["comments"] + eps)
    features["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    features["engagement_per_char"] = features["total_engagement"] / (df["character_count"] + eps)

    # Content density signals
    # features["hashtag_density"] = df["hashtag_count"] / (df["character_count"] + eps)
    # features["mention_density"] = df["mention_count"] / (df["character_count"] + eps)
    # features["url_per_char"] = df["contains_url"] / (df["character_count"] + eps)

    # Log transforms to reduce skew on count columns
    features["log_likes"] = np.log1p(df["likes"])
    features["log_comments"] = np.log1p(df["comments"])
    features["log_shares"] = np.log1p(df["shares"])
    features["log_character_count"] = np.log1p(df["character_count"])

    # Lexical diversity from post content using Type-Token Ratio
    if "content" in df.columns:
        features["lexical_diversity"] = compute_lexical_diversity(df["content"])

    return features


def scale_features(X_df):
    scaler = StandardScaler()
    return scaler.fit_transform(X_df)


def train_isolation_forest(X_train, X_all, contamination=0.5):
    model = IsolationForest(n_estimators=300, contamination=contamination, max_features=1.0, random_state=42)
    model.fit(X_train)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return model, predictions, scores


def train_lof(X_train, X_all, contamination=0.5):
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=contamination)
    model.fit(X_train)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)

    return predictions, scores


# Derives feature importance from IsolationForest by counting how many time 
# each feature is used as a split node across all trees, the more it splits, the more useeful the features is.
def plot_feature_importance(if_model, feature_names, title="Feature_Importance_IF"):
    n_features = len(feature_names)
    importances = np.zeros(n_features)

    for tree in if_model.estimators_:
        for feature_idx in tree.tree_.feature:
            if feature_idx >= 0:
                importances[feature_idx] += 1

    total = importances.sum()
    if total > 0:
        importances /= total

    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    plt.figure(figsize=(8, max(4, n_features * 0.35)))
    plt.barh(sorted_names, sorted_vals, color="#4f6ef7")
    plt.xlabel("Relative Importance (split frequency)")
    plt.title(title.replace("_", " "))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_post.png"))
    plt.close()

    print("\n")
    print("Print top 5 imporantce features: ")
    for name, val in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]:
        print(f"{name}: {val:.4f}")


def plot_anomaly_results(predictions, title):
    normal = (predictions == 1).sum()
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_post.png"))
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
    print("\n")

    return accuracy, precision, recall, f1


def main(df, X_pca=None):
    # Change the is fake from Boolean to Integer
    y_true = df["is_fake"].astype(bool).astype(int).values

    X_eng = engineer_features(df)
    X_scaled = scale_features(X_eng)

    CONTAMINATION = 0.5

    X_normal = X_scaled[~df["is_fake"].astype(bool)]

    # Train Isolation Forest model
    if_model, if_preds, if_score = train_isolation_forest(X_normal, X_scaled, CONTAMINATION)

    print("\nIsolation Forest raw predictions")
    print(pd.Series(if_preds).value_counts())
    plot_anomaly_results(if_preds, "Isolation_Forest_Result")
    plot_feature_importance(if_model, list(X_eng.columns), "Feature_Importance_IF")

    # Convert raw predictions where -1 (anomaly) means 1 (fake), and 1 (normal) means 0 (real)
    if_y_pred = (if_preds == -1).astype(int)
    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    evaluate_model(y_true, if_y_pred, "Isolation Forest")

    # Train the Local Outlier Factor model
    lof_preds, lof_score = train_lof(X_normal, X_scaled, CONTAMINATION)

    print("\nLOF raw predictions")
    print(pd.Series(lof_preds).value_counts())
    plot_anomaly_results(lof_preds, "LOF_Result")

    # Convert raw predictions whre -1 (anomaly) means 1 (fake), and 1 (normal) means 0 (real)
    lof_y_pred = (lof_preds == -1).astype(int)
    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return if_preds, lof_preds

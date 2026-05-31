import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew
import json
from sklearn.preprocessing import StandardScaler


# Get the cleaned dataset path
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "post")
# Get the PCA dataset path
PCA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "pca_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the datzet
def load_dataset():
    df = pd.read_csv(CLEAN_PATH)
    return df


# Caluclate the lexical diversity score by using the Type-Token RAtio (unique words / total words)
def compute_lexical_diversity(text_input):
    eps = 1e-6
    def ttr(text):
        if pd.isna(text) or str(text).strip() == "":
            return 0.0
        words = str(text).lower().split()
        return len(set(words)) / (len(words) + eps)

    if isinstance(text_input, pd.Series):
        return text_input.apply(ttr)
    return ttr(text_input)


def engineer_features(df):
    eps = 1e-6
    features = pd.DataFrame(index=df.index)

    # Exclude label-encoded categoricals and zero-variance columns
    encoded_categoricals = {
        "is_fake", "activity_id", "user_id",
        "post_country", "post_region", "post_city",
        "device", "platform", "media_type", "content_type", "language", "contains_url",
        "has_media", "is_weekend", "day_of_week", "hour_of_day", "hashtag_count",
        "likes", "shares", "comments", "character_count",
        "mention_count",  # all zeros in this dataset
    }

    # Get all numerical columns
    numerical_columns = (df.select_dtypes(include='number').columns)

    # Raw numeric signals (skip encoded categoricals)
    for col in numerical_columns:
        if col in df.columns and col not in encoded_categoricals:
            features[col] = df[col].astype(float)

    # Engagement ratios
    features["comments_per_like"] = np.log1p(df["comments"] / (df["likes"] + 1))
    features["shares_per_like"] = np.log1p(df["shares"] / (df["likes"] + 1))
    features["shares_per_comment"] = np.log1p(df["shares"] / (df["comments"] + 1))
    features["total_engagement"] = np.log1p(df["likes"] + df["comments"] + df["shares"])
    features["engagement_per_char"] = np.log1p((df["likes"] + df["comments"] + df["shares"]) / (df["character_count"] + 1))

    # Content density signals
    features["hashtag_density"] = np.log1p(df["hashtag_count"] / (df["character_count"] + 1))
    features["url_per_char"] = np.log1p(df["contains_url"] / (df["character_count"] + 1))

    # Lexical diversity from post content using Type-Token Ratio
    if "content" in df.columns:
        features["lexical_diversity"] = compute_lexical_diversity(df["content"])

    # Log transforms to reduce skew on count columns
    features["log_likes"] = np.log1p(df["likes"])
    features["log_comments"] = np.log1p(df["comments"])
    features["log_shares"] = np.log1p(df["shares"])
    features["log_character_count"] = np.log1p(df["character_count"])

    return features


# Select all numerical features
def select_features(df):
    numerical_columns = df.select_dtypes(include='number').columns

    drop_columns = {"is_fake", "activity_id", "user_id", "timestamp", "content"}

    importance_features = [col for col in numerical_columns if col not in drop_columns]

    X = df[importance_features]

    print("\nSelected Features for PCA:")
    print(importance_features)

    return X


# Normalizes the features
def normalize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# Apply PCA to the scaled X
def apply_pca(X_scaled):
    pca = PCA(n_components=0.95, random_state=42)

    X_pca = pca.fit_transform(X_scaled)

    print("\n")
    print("Original Features:", X_scaled.shape[1])
    print("Reduced Features:", X_pca.shape[1])
    print("Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    print("\nTotal Explained Variance:")
    print(pca.explained_variance_ratio_.sum())

    return pca, X_pca


# Plot a scatter plot to visualize the PCA, colored by performance label
def plot_pca_scatter(X_pca, df):

    plt.figure(figsize=(10, 6))

    labels = df["is_fake"].values
    unique_labels = [1, 0]
    colors = ["#d62728", "#2ca02c"]

    if X_pca.shape[1] < 2:
        print("PCA produced only 1 component — skipping 2D scatter plot.")
        plt.close()
        return

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, label=label, color=color, s=10)


    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection (colored by REAL vs FAKE Post)")
    plt.legend(title="Performance", markerscale=3)
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_projection_post.png"), dpi=150)
    plt.close()



def plot_explained_variance(pca):
    cumulative_variance = (pca.explained_variance_ratio_.cumsum())

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_DIR,"pca_variance_post.png"))

    plt.close()


def perform_PCA(df):
    # X = select_features(df)
    X_eng = engineer_features(df)
    print("Engineer fearures for post: ")
    print(X_eng.columns)
    X_scaled = normalize_features(X_eng)
    pca, X_pca = apply_pca(X_scaled)

    print("\n")
    # print("X_PCA:")
    # print(X_pca)

    plot_pca_scatter(X_pca, df)
    plot_explained_variance(pca)

    print("Task 3 done.")

    return pca, X_pca



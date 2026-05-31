import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.stats import skew
import json

# Get Raw Data from File Path
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "Instagram_fake_profile_dataset.csv")
# The csv file path after data cleaning
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_account_data.csv")
# Get the output directory
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "account")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the datzet
def load_dataset():
    df = pd.read_csv(CLEAN_PATH)
    return df


# Select all numerical features
def select_features_account(df):
    
    numerical_columns = (df.select_dtypes(include='number').columns)
    # Columns to remove
    drop_columns = ["fake"]

    # Keep only useful features
    features = [col for col in numerical_columns if col not in drop_columns]

    X = df[features]

    print("\nSelected Features:")
    print(features)

    return X


def engineer_features(df):
    # eps = 1e-6
    features = pd.DataFrame(index=df.index)

    numerical_columns = (df.select_dtypes(include='number').columns)
    # Columns to remove
    drop_columns = ["fake", "nums/length fullname", "external URL", "private", 
                    "name==username", "#posts", "#followers", '#follows', "description length"]

    # Keep only useful features
    important_features = [col for col in numerical_columns if col not in drop_columns]

    # Raw profile attributes
    for col in important_features:
        if col in df.columns:
            features[col] = df[col]

    # Enginering features
    features["log_follower_following_ratio"] = np.log1p(df["#followers"] / (df["#follows"] + 1))
    # features["posts_per_follower"] = df["#posts"] / (df["#followers"] + 1)
    # features["following_per_follower"] = df["#follows"] / (df["#followers"] + 1)
    # features["posts_per_following"] = df["#posts"] / (df["#follows"] + 1)

    features["log_description"] = np.log1p(df["description length"])
    features["has_description"] = (df["description length"] > 0).astype(int)

    features["log_posts"] = np.log1p(df["#posts"])
    features["log_followers"] = np.log1p(df["#followers"])
    features["log_follows"] = np.log1p(df["#follows"])

    # features["username_nums_sq"] = df["nums/length username"] ** 2
    # features["fullname_nums_sq"] = df["nums/length fullname"] ** 2

    return features


# Normalizes the features
def normalize_features_account(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# Apply PCA to the scaled X
def apply_pca_account(X_scaled):
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
def plot_pca_scatter_account(X_pca, df):

    plt.figure(figsize=(10, 6))

    labels = df["fake"].values
    unique_labels = [1, 0]
    colors = ["#d62728", "#2ca02c"]

    if X_pca.shape[1] < 2:
        # Only 1 component — plot as 1D strip chart
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], np.zeros(mask.sum()), alpha=0.4, label=label, color=color, s=10)
        plt.xlabel("Principal Component 1")
        plt.ylabel("")
        plt.yticks([])
    else:
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, label=label, color=color, s=10)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

    plt.title("PCA Projection (colored by performance bucket)")
    plt.legend(title="Performance", markerscale=3)
    plt.savefig(os.path.join(OUTPUT_PATH, "pca_projection_account.png"), dpi=150)
    plt.show()



def plot_explained_variance_account(pca):
    cumulative_variance = (pca.explained_variance_ratio_.cumsum())

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_PATH,"pca_variance_account.png"))

    plt.show()


def perform_PCA_account(df):
    # X = select_features_account(df)
    X_eng = engineer_features(df)
    print("Engineer fearures for account: ")
    print(X_eng.columns)
    X_scaled = normalize_features_account(X_eng)
    pca, X_pca = apply_pca_account(X_scaled)

    print("\n")
    # print("X_PCA:")
    # print(X_pca)

    plot_pca_scatter_account(X_pca, df)
    plot_explained_variance_account(pca)

    print("Account Detect Task 3 done.")

    return pca, X_pca


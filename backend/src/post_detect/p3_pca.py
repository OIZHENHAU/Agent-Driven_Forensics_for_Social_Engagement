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


# Select all numerical features
def select_features(df):
    
    numerical_columns = (df.select_dtypes(include='number').columns)
    # Columns to remove
    drop_columns = ["account_id", "post_id"]

    # Keep only useful features
    features = [col for col in numerical_columns if col not in drop_columns]

    X = df[features]

    print("\nSelected Features:")
    print(features)

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

    labels = df["performance_bucket_label"].values
    unique_labels = ["low", "medium", "high", "viral"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, label=label, color=color, s=10)


    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection (colored by performance bucket)")
    plt.legend(title="Performance", markerscale=3)
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_projection_post.png"), dpi=150)
    plt.show()



def plot_explained_variance(pca):
    cumulative_variance = (pca.explained_variance_ratio_.cumsum())

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_DIR,"pca_variance_post.png"))

    plt.show()


def perform_PCA(df):
    X = select_features(df)
    X_scaled = normalize_features(X)

    pca, X_pca = apply_pca(X_scaled)

    print("\n")
    # print("X_PCA:")
    # print(X_pca)

    plot_pca_scatter(X_pca, df)
    plot_explained_variance(pca)

    print("Task 3 done.")

    return pca, X_pca



# Save the PCA-transformed dataset
'''def save_pca_data(X_pca, y: pd.Series, n_components: int):
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca["is_fake"] = y.values
    os.makedirs(os.path.dirname(PCA_PATH), exist_ok=True)
    df_pca.to_csv(PCA_PATH, index=False)
    print(f"PCA data saved to: {PCA_PATH}")


def save_feature_json(features):
    meta = {"features": features}
    meta_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "pca_features.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print("Importance faeture saved successfully.")
'''

'''
def main():
    df = load_dataset()
    X = select_features(df)
    X_scaled = normalize_features(X)

    pca, X_pca = apply_pca(X_scaled)

    print("\n")
    # print("X_PCA:")
    # print(X_pca)

    plot_pca_scatter(X_pca, df)
    plot_explained_variance(pca)


if __name__ == "__main__":
    main()

'''

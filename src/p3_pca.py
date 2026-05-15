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
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
# Get the PCA dataset path
PCA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pca_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour to determine the real or fake account, grren represent real and red represent fake.
PALETTE = {0: "#2ecc71", 1: "#e74c3c"}

#Get the cleaned dataset
def load_dataset() -> pd.DataFrame:
    clean_dataset = pd.read_csv(CLEAN_PATH)
    return clean_dataset


# Check the skewness for each columns in the datsset
def get_skewness():
    df = load_dataset()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].apply(skew).sort_values(ascending=False)
    print(skewness)


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


#Scale the features
def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Fit PCA and plot explained variance using scree plot
def plot_scree(X_scaled, n_components=None):
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Individual explained variance
    axes[0].bar(range(1, len(explained) + 1), explained, color="steelblue")
    axes[0].set_title("Explained Variance per Component")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")

    # Cumulative explained variance
    axes[1].plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="steelblue")
    axes[1].axhline(0.90, color="red", linestyle="--", label="90% threshold")
    axes[1].axhline(0.95, color="orange", linestyle="--", label="95% threshold")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_scree_plot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Scree plot saved to: {path}")

    # Print how many components needed for 90% and 95%
    n90 = np.argmax(cumulative >= 0.90) + 1
    n95 = np.argmax(cumulative >= 0.95) + 1
    print(f"Components needed for 90% variance: {n90}")
    print(f"Components needed for 95% variance: {n95}")

    return pca_full, n90


# Apply PCA with chosen number of components.
def apply_pca(X_scaled, n_components: int):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca


# Plot PC1 vs PC2 coloured by Real & Fake
def plot_pca_scatter(X_pca, y: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, name in [(0, "Real"), (1, "Fake")]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[label], label=name, alpha=0.4, s=15)

    ax.set_title("PCA PC1 vs PC2 (Real vs Fake)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"PCA scatter plot saved to: {path}")


# Plot top feature loadings for PC1 and PC2
def plot_loadings(pca, feature_names: list, top_n: int = 10):
    loadings = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=["PC1", "PC2"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, pc in enumerate(["PC1", "PC2"]):
        top = loadings[pc].abs().nlargest(top_n).index
        data = loadings.loc[top, pc].sort_values()
        colors = ["#e74c3c" if v < 0 else "lightblue" for v in data]
        axes[i].barh(data.index, data.values, color=colors)
        axes[i].axvline(0, color="black", linewidth=0.8)
        axes[i].set_title(f"Top {top_n} Feature Loadings – {pc}")
        axes[i].set_xlabel("Loading")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_loadings.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"PCA loadings plot saved to: {path}")



# Save the PCA-transformed dataset
def save_pca_data(X_pca, y: pd.Series, n_components: int):
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca["is_fake"] = y.values
    os.makedirs(os.path.dirname(PCA_PATH), exist_ok=True)
    df_pca.to_csv(PCA_PATH, index=False)
    print(f"PCA data saved to: {PCA_PATH}")


def save_feature_json(features):
    meta = {"features": features}
    meta_path = os.path.join(os.path.dirname(__file__), "..", "data", "pca_features.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print("Importance faeture saved successfully.")


def main():
    X, y, feature_names = load_and_prepare_data()
    # we Scale the features first
    X_scaled = scale_features(X)
    # We do scen plot to gather the number of components need to fit our PCA
    X_scalee, n_components = plot_scree(X_scaled)
    # We apply PCA with chosen components with varaince more than or equal to 90%
    pca, X_pca = apply_pca(X_scaled, n_components)
    # Do scatter plot between real and fake account
    plot_pca_scatter(X_pca, y)
    # We do feature loadings plot
    plot_loadings(pca, feature_names)
    # Save PCA dataset
    save_pca_data(X_pca, y, n_components)
    #Save the importnace faetures into the pca_meta.json file
    save_feature_json(feature_names)


if __name__ == "__main__":
    main()

import os, sys, json, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# Get the path for cleaned datasert
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)
# Green means real, Red means fake
PALETTE = {0: "#2ecc71", 1: "#e74c3c"}


# Load the cleaned dataset
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    return df

def basic_statistic_view(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBasic statistics:")
    print(df.describe(include="all").T.to_string())
    print("\nClass balance:")
    print(df["is_fake"].value_counts(normalize=True).mul(100).round(2)
          .rename(index={0: "Real", 1: "Fake"}).to_string(), "%")
    

def plot_distribution(df: pd.DataFrame) -> None:
    num_columns = ["followers", "following", "follower_following_ratio",
                   "posts", "account_age_days", "suspicious_engagement_rate",
                   "spam_comments_rate", "generic_comment_rate"]
    
    # Create a figure of 4 * 2 grid of subplots and set the overall suize of width 14px and height 16px
    fig, axes = plt.subplots(4, 2, figsize = (14, 16))
    # Convert the 2D array into 1D array
    axes = axes.ravel()
    for index, column in enumerate(num_columns):
        ax = axes[index]
        #print("Column ", column)

        for label, group in df.groupby("is_fake"):
            # print("Group ", group)
            value = group[column].dropna()
            value = value[value > 0]

            if value.empty:
                continue

            if label == 1:
                hist_label = "Fake"
            else:
                hist_label = "Real"

            # Use log1p instead of log because it handles zero value safely
            log_vals = np.log1p(value)
            sns.histplot(log_vals, ax=ax, kde=True, color=PALETTE[label], label=hist_label, alpha=0.55, bins=40)
        
        ax.set_title(f"{column}")
        ax.set_xlabel("")
        ax.legend()
    
    fig.suptitle("Histogram Plot Distributions by Account Type", fontsize=15)
    path = os.path.join(OUTPUT_PATH, "histogram_plot_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Histgram plot saved to path: {path}")


# df = load_dataset()
# plot_distribution(df)

def plot_correlation_diagram(df: pd.DataFrame) -> pd.DataFrame:
    column_integer = df.select_dtypes(include=[np.number]).drop(columns=["is_fake"])
    # Calculate the Pearson correlation between every numeric column
    corr = column_integer.corr()

    fig, axes = plt.subplots(figsize=(14, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes)
    axes.set_title("Feature Correlation Matrix", fontsize=15)
    # Give th plot more padding space around the figure
    plt.tight_layout()
    path = os.path.join(OUTPUT_PATH, "correlation_plot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Correlation plot saved to: {path}")


# BoxPlot for comparing each features based on real and fake account
def plot_boxplots(df: pd.DataFrame) -> None:
    key_features = ["followers", "following", "follower_following_ratio", "suspicious_engagement_rate", "spam_comments_rate",
                    "caption_similarity_score", "content_similarity_score", "username_randomness"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.ravel()

    for index, column in enumerate(key_features):
        axe = axes[index]
        
        plot_is_fake = df[["is_fake", column]].copy()
        plot_is_fake["is_fake"] = plot_is_fake["is_fake"].map({0: "Real", 1: "Fake"})
        plot_is_fake[column] = np.log1p(plot_is_fake[column].clip(lower=0))

        # Create box plot
        sns.boxplot(data=plot_is_fake, x="is_fake", y=column, hue="is_fake", palette={"Real": PALETTE[0], "Fake": PALETTE[1]}, ax=axe)
        
        axe.set_title(f"{column}")
        axe.set_xlabel("")

    fig.suptitle("Feature Distributions: Real vs Fake Accounts", fontsize=15)
    fig.tight_layout()
    

    path = os.path.join(OUTPUT_PATH, "box_plot_features.png")
    fig.savefig(path)
    plt.close(fig)
    
    print(f"Box plot (Real vs Fake) saved to: {path}")


# df = load_dataset()
# plot_distribution(df)
# plot_correlation_diagram(df)
# plot_boxplots(df)

# Plot a diagram based on the platform user use with chat form and histogram to determine the percentage of fake
def platform_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    platform_counts = df["platform"].value_counts()
    # axes[0].pie(platform_counts, labels=platform_counts.index, autopct="%1.1f%%", startangle=140)
    axes[0].pie(platform_counts, labels=platform_counts.index, autopct="%1.1f%%")
    axes[0].set_title("Platform Distribution")

    platform_fake = (df.groupby("platform")["is_fake"].mean().mul(100).sort_values(ascending=False))
    sns.barplot(x=platform_fake.index, y=platform_fake.values,palette="Reds_d", ax=axes[1])

    axes[1].set_title("Fake Account Rate Based on Platform (%)")
    axes[1].set_ylabel("Fake %")
    axes[1].set_xlabel("")

    fig.tight_layout()
    path = os.path.join(OUTPUT_PATH, "platform_plot.png")
    fig.savefig(path)
    plt.close(fig)

    print(f"Platform plot successfully saved to path: {path}")


# df = load_dataset()
# platform_plot(df)


'''def main():
    df = load_dataset()
    basic_statistic_view(df)
    plot_distribution(df)
    plot_correlation_diagram(df)
    plot_boxplots(df)
    platform_plot(df)


if __name__ == "__main__":
    main()
'''

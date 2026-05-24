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
from scipy.stats import skew

# Get Raw Data from File Path
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Instagram_Analytics.csv")
# The csv file path after data cleaning
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(OUTPUT_PATH, exist_ok=True)


# Load the cleaned dataset
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    return df


#Plot the histogram distribution
def plot_histogram(df: pd.DataFrame) -> None:

    numeric_columns = df.select_dtypes(include='number').columns

    num_features = len(numeric_columns)
    rows = (num_features // 3) + 1

    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))

    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):

        axes[i].hist(df[col], bins=30)

        axes[i].set_title(f"Distribution of {col}")

        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")


    # Remove unused subplot
    for j in range(len(numeric_columns),len(axes)):
        fig.delaxes(axes[j])


    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH,"all_histograms_features.png"))

    plt.show()


# Plot all boxplot for all numerical columns
def plot_boxplots(df: pd.DataFrame) -> None:

    numeric_columns = df.select_dtypes(include='number').columns
    num_features = len(numeric_columns)
    rows = (num_features // 3) + 1

    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):

        axes[i].boxplot(df[col])

        axes[i].set_title(f"Boxplot of {col}")

    # Remove unused subplot
    for j in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[j])


    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "all_boxplots_features.png"))

    plt.show()


# PLlot correlation heatmap diagram
def correlation_heatmap(df):

    numeric_columns = df.select_dtypes(include=np.number)

    correlation_matrix = numeric_columns.corr()

    plt.figure(figsize=(14, 8))

    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")

    plt.title("Correlation Heatmap")
    plt.savefig("backend/outputs/correlation_heatmap.png")

    plt.show()


# Plot a log validation diagram
def validate_log_normal(df: pd.DataFrame) -> None:

    numeric_columns = (df.select_dtypes(include=np.number).columns)

    valid_columns = []
    skipped_columns = []

    # Check which columns are safe
    for col in numeric_columns:

        min_value = df[col].min()

        if min_value > -1:
            valid_columns.append(col)
        else:
            skipped_columns.append(col)

    print("\nSafe Columns:")
    print(valid_columns)

    print("\nSkipped Columns:")
    print(skipped_columns)

    # Create subplot
    num_features = len(valid_columns)
    rows = (num_features // 3) + 1

    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))

    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        clean_data = (df[col].dropna())

        original_skew = skew(clean_data)
        log_data = np.log1p(clean_data)
        transformed_skew = skew(log_data)

        print( f"\n{col}")

        print(f"Original Skewness: " f"{original_skew:.2f}")

        print(f"Log Transformed " f"Skewness: " f"{transformed_skew:.2f}")

        axes[i].hist(log_data, bins=30)

        axes[i].set_title(f"Log Distribution\n{col}")


    # Remove empty plots
    for j in range(len(valid_columns), len(axes)):
        fig.delaxes(axes[j])


    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH,"all_log_validation_features.png"))

    plt.show()


def exploratory_analysis(df):
    plot_histogram(df)
    plot_boxplots(df)
    correlation_heatmap(df)
    validate_log_normal(df)

    print("Task 2 done.")


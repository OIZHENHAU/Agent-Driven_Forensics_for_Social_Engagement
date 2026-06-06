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
from scipy.stats import skew
from src.post_detect.p3_pca import (apply_pca, plot_pca_scatter, normalize_features, engineer_features)


CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "post")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Plot a log validation diagram
def validate_log_engineer_features(df: pd.DataFrame) -> None:

    numeric_columns = (df.select_dtypes(include=np.number).columns)

    valid_columns = []
    skipped_columns = []

    # Check which columns are safe watch out for non-zero variance for skewness
    for col in numeric_columns:
        min_value = df[col].min()

        if min_value > -1 and df[col].std() > 0:
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
    plt.savefig(os.path.join(OUTPUT_DIR, "all_log_validation_features_engineer_post.png"))

    plt.show()


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


# Circular process of checking the enginnering features
def check_enginner_features(df):
    X_eng = engineer_features(df)
    validate_log_engineer_features(X_eng)
    X_scaled = normalize_features(X_eng)
    pca, X_pca = apply_pca(X_scaled)
    plot_pca_scatter(X_pca, df)


def main(df, X_pca=None):
    # Change the is fake from Boolean to Integer
    y_true = df["is_fake"].astype(bool).astype(int).values
    X_scaled = X_pca

    CONTAMINATION = 0.5

    X_normal = X_scaled[~df["is_fake"].astype(bool)]
    # X_normal = X_pca

    if_model, if_preds, if_score = train_isolation_forest(X_normal, X_scaled, CONTAMINATION)
    if_y_pred = (if_preds == -1).astype(int)
    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    if_accuracy, if_precision, if_recall, if_f1 = evaluate_model(y_true, if_y_pred, "Isolation Forest")

    lof_preds, lof_score = train_lof(X_normal, X_scaled, CONTAMINATION)
    lof_y_pred = (lof_preds == -1).astype(int)
    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    lof_accuracy, lof_precision, lof_recall, lof_f1 = evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return {
        "isolation_forest": {
            "accuracy": round(float(if_accuracy), 4),
            "precision": round(float(if_precision), 4),
            "recall": round(float(if_recall), 4),
            "f1": round(float(if_f1), 4),
        },
        "lof": {
            "accuracy": round(float(lof_accuracy), 4),
            "precision": round(float(lof_precision), 4),
            "recall": round(float(lof_recall), 4),
            "f1": round(float(lof_f1), 4),
        }
    }
    

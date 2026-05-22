import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)


# Get the cleaned dataset path
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Train the IsolationForest model
def train_isolation_forest(X, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)

    return predictions

# Train the LOF model
def train_lof(X, contamination=0.05):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    predictions = model.fit_predict(X)

    return predictions


# Visualize anomaly distribution
def plot_anomaly_results(predictions, title):

    normal = (predictions == 1).sum()
    anomaly = (predictions == -1).sum()

    labels = ["Normal", "Anomaly"]
    values = [normal, anomaly]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)

    plt.title(title)
    plt.ylabel("Count")

    save_path = os.path.join(OUTPUT_DIR,f"{title}.png")

    plt.savefig(save_path)
    plt.close()


# Plot the confusion matrix of the to model by convert model output if IsolationForest / LOF: to -1 = anomaly & 1 = normal
def plot_confusion_matrix(y_true, predictions, title):
    y_pred = np.where(predictions == -1, 1, 0)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm,annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    save_path = os.path.join(OUTPUT_DIR,f"{title}.png")

    plt.savefig(save_path)
    plt.close()


# Evaluate the model's accuracy, precision, and recall
def evaluate_model(y_true, predictions, model_name):
    y_pred = np.where(predictions == -1, 1, 0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n######## {model_name} Evaluation ##########")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n")


def main(X, df):
    # pseudo labels
    y_true = np.where(df["performance_bucket_label"] == "viral", 1, 0)

    # Match contamination to actual viral proportion so models flag the right number of anomalies
    viral_rate = y_true.mean()

    # Isolation Forest Details
    if_predictions = train_isolation_forest(X, contamination=viral_rate)

    print("\nIsolation Forest Results")
    print(pd.Series(if_predictions).value_counts())

    plot_anomaly_results(if_predictions, "Isolation_Forest_Result")

    plot_confusion_matrix(y_true, if_predictions, "Isolation_Forest_Confusion_Matrix")

    evaluate_model(y_true, if_predictions, "Isolation Forest")


    # LOF Details
    lof_predictions = train_lof(X, contamination=viral_rate)

    print("\nLOF Results")
    print(pd.Series(lof_predictions).value_counts())

    plot_anomaly_results(lof_predictions, "LOF_Result")

    plot_confusion_matrix(y_true, lof_predictions, "LOF_Confusion_Matrix")

    evaluate_model(y_true, lof_predictions, "Local Outlier Factor")


    return if_predictions, lof_predictions



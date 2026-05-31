import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from src.account_detect.p3_fake_account_pca import (engineer_features, normalize_features_account, apply_pca_account, plot_pca_scatter_account)


CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_account_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "account")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_isolation_forest(X_train, X_all, contamination=0.5):
    model = IsolationForest(n_estimators=300, contamination=contamination, max_features=1.0, random_state=42)
    model.fit(X_train)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


def train_lof(X_train, X_all, contamination=0.5):
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=contamination)
    model.fit(X_train)
    scores = model.decision_function(X_all)
    predictions = model.predict(X_all)
    return predictions, scores


def plot_anomaly_results(predictions, title):
    real = (predictions ==  1).sum()
    fake = (predictions == -1).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], [real, fake])
    plt.title(title)
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.close()


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n######## {model_name} Evaluation ##########")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n")

    return accuracy, precision, recall, f1


# Use to calculate the authentic score of the model
def auth_score(value, ref):
    return int(np.clip(np.mean(ref <= value) * 100, 1, 100))


# Train all the IsoaltionForest and LOF model on the cleaned dataset
def predict_single_account(raw_input: dict) -> dict:
    df = pd.read_csv(CLEAN_PATH)

    single_row = pd.DataFrame([{
        "profile pic": int(raw_input.get("profile_pic", 0)),
        "nums/length username": float(raw_input.get("nums_length_username", 0)),
        "fullname words": int(raw_input.get("fullname_words", 1)),
        "nums/length fullname": float(raw_input.get("nums_length_fullname", 0)),
        "name==username": int(raw_input.get("name_equals_username", 0)),
        "description length": int(raw_input.get("description_length", 0)),
        "external URL": int(raw_input.get("external_url", 0)),
        "private": int(raw_input.get("private", 0)),
        "#posts": int(raw_input.get("posts", 0)),
        "#followers": int(raw_input.get("followers", 0)),
        "#follows": int(raw_input.get("follows", 0)),
    }])

    # real_mask = (df["fake"] == 0).values
    df_features = df.drop(columns=["fake"])
    df_combined = pd.concat([df_features, single_row], ignore_index=True)

    X_eng = engineer_features(df_combined)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_eng)
    account_pca, X_pca = apply_pca_account(X_scaled)

    CONTAMINATION = 0.5
    X_ref = X_pca[:len(df)]
    # X_train = X_ref[real_mask]
    X_train = X_ref

    if_model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, max_features=1.0, random_state=42)
    if_model.fit(X_train)
    if_scores = if_model.decision_function(X_pca)
    if_ref_scores = if_model.decision_function(X_train)


    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=CONTAMINATION)
    lof_model.fit(X_train)
    lof_train_scores = lof_model.decision_function(X_train)
    lof_ref_scores = float(lof_model.decision_function(X_pca[-1:])[0])

    if_auth = auth_score(if_scores[-1], if_ref_scores)
    lof_auth = auth_score(lof_train_scores, lof_ref_scores)
    ensemble = int(round((if_auth + lof_auth) / 2))

    return {"if_score": if_auth, "lof_score": lof_auth, "ensemble_score": ensemble, "verdict": "Authentic" if ensemble >= 50 else "Suspicious"}


ACCOUNT_CSV_COLUMN_MAP = {
    "followers": "#followers", 
    "follower_count": "#followers",
    "following": "#follows", 
    "follows": "#follows", 
    "follow_count": "#follows",
    "posts": "#posts", 
    "post_count": "#posts",
    "profile_pic": "profile pic", 
    "has_profile_pic": "profile pic",
    "nums_length_username": "nums/length username", 
    "username_digit_ratio": "nums/length username",
    "fullname_words": "fullname words", 
    "fullname_word_count": "fullname words",
    "nums_length_fullname": "nums/length fullname", 
    "fullname_digit_ratio": "nums/length fullname",
    "name_equals_username": "name==username", 
    "name_eq_username": "name==username",
    "description_length": "description length", 
    "bio_length": "description length",
    "external_url": "external URL", 
    "has_external_url": "external URL",
    "is_private": "private",
}

ACCOUNT_COLUMN_USE = ["profile pic", "nums/length username", "fullname words", "nums/length fullname",
                    "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"]


def predict_batch_accounts(rows: list) -> list:
    df = pd.read_csv(CLEAN_PATH)

    def normalize_dataset_row(row):
        csv_output = {}
        for key, value in row.items():
            key = key.strip()
            mapped = ACCOUNT_CSV_COLUMN_MAP.get(key.lower(), ACCOUNT_CSV_COLUMN_MAP.get(key, key))
            csv_output[mapped] = value

        result = {}
        for column in ACCOUNT_COLUMN_USE:
            try:
                result[column] = float(csv_output.get(column, 0))

            except (ValueError, TypeError):
                result[column] = 0.0

        return result

    # real_mask = (df["fake"] == 0).values
    input_rows = [normalize_dataset_row(r) for r in rows]
    input_df = pd.DataFrame(input_rows)

    df_features = df.drop(columns=["fake"])
    df_combined = pd.concat([df_features, input_df], ignore_index=True)

    X_eng = engineer_features(df_combined)
    X_scaled = RobustScaler().fit_transform(X_eng)
    account_pca, X_pca = apply_pca_account(X_scaled)

    CONTAMINATION = 0.5
    X_ref = X_pca[:len(df)]
    # X_train = X_ref[real_mask]
    X_train = X_ref

    if_model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, max_features=1.0, random_state=42)
    if_model.fit(X_train)
    if_scores_all = if_model.decision_function(X_pca)
    if_ref_scores = if_model.decision_function(X_train)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=CONTAMINATION)
    lof_model.fit(X_train)
    lof_train_scores = lof_model.decision_function(X_train)
    lof_input_scores = lof_model.decision_function(X_pca[len(df):])

    all_row_results = []
    for i, row in enumerate(rows):
        index = len(df) + i
        if_auth = auth_score(if_scores_all[index], if_ref_scores)
        lof_auth = auth_score(lof_input_scores[i], lof_train_scores)
        ensemble = int(round((if_auth + lof_auth) / 2))
        all_row_results.append({"if_score": if_auth, "lof_score": lof_auth, "ensemble_score": ensemble, 
                                "verdict": "Authentic" if ensemble >= 50 else "Suspicious", "row_data": row})

    return all_row_results


# Plot a log validation diagram
def validate_log_engineer_account_features(df: pd.DataFrame) -> None:

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


# Circular process of checking the enginnering features
def check_enginner_accoount_features(df):
    X_eng = engineer_features(df)
    validate_log_engineer_account_features(X_eng)
    X_scaled = normalize_features_account(X_eng)
    pca, X_pca = apply_pca_account(X_scaled)
    plot_pca_scatter_account(X_pca, df)


def main(df, X_pca=None):
    y_true = df["fake"].values

    X_scaled = X_pca

    contamination = 0.5
    X_normal = X_scaled[(df["fake"] == 0).values]
    # X_normal = X_scaled

    # Train Isolation Forest
    if_preds, if_anomaly_score = train_isolation_forest(X_normal, X_scaled, contamination=contamination)

    print("\nIsolation Forest raw predictions")
    print(pd.Series(if_preds).value_counts())
    plot_anomaly_results(if_preds, "Isolation_Forest_Result")

    if_y_pred = (if_preds == -1).astype(int)
    plot_confusion_matrix(y_true, if_y_pred, "Isolation_Forest_Confusion_Matrix")
    evaluate_model(y_true, if_y_pred, "Isolation Forest")

    # Train Local Outlier Factor
    lof_preds, lof_anomaly_score = train_lof(X_normal, X_scaled, contamination)

    print("\nLOF raw predictions")
    print(pd.Series(lof_preds).value_counts())
    plot_anomaly_results(lof_preds, "LOF_Result")

    lof_y_pred = (lof_preds == -1).astype(int)
    plot_confusion_matrix(y_true, lof_y_pred, "LOF_Confusion_Matrix")
    evaluate_model(y_true, lof_y_pred, "Local Outlier Factor")

    return if_preds, lof_preds
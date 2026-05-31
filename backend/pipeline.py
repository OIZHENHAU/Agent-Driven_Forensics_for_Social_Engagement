import json
import os

from src.post_detect.p1_data_cleaning import clean_dataset
from src.post_detect.p2_eda import exploratory_analysis
from src.post_detect.p3_pca import perform_PCA
from src.post_detect.p4_ml_agent import main as train_model_post
from src.post_detect.p4_ml_agent import (check_enginner_features)

from src.account_detect.p1_fake_account_cleaning import clean_dataset_account
from src.account_detect.p2_fake_account_eda import exploratory_analysis_account
from src.account_detect.p3_fake_account_pca import perform_PCA_account
from src.account_detect.p4_fake_account_detection import main as training_model_account
from src.account_detect.p4_fake_account_detection import check_enginner_accoount_features

MODEL_RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model_result.json")


def main():
    # Post Detection
    cleaned_df = clean_dataset()
    exploratory_analysis(cleaned_df)
    post_pca, X_post_pca = perform_PCA(cleaned_df)
    post_model_results = train_model_post(cleaned_df, X_pca=X_post_pca)

    # Account Detection
    cleaned_account_df = clean_dataset_account()
    exploratory_analysis_account(cleaned_account_df)
    account_pca, X_account_pca = perform_PCA_account(cleaned_account_df)
    account_model_results = training_model_account(cleaned_account_df, X_pca=X_account_pca)

    # Save metrics to JSON for Dashboard 1
    all_model_result = {
        "post_detection": post_model_results,
        "account_detection": account_model_results,
    }

    os.makedirs(os.path.dirname(MODEL_RESULT_PATH), exist_ok=True)
    with open(MODEL_RESULT_PATH, "w") as f:
        json.dump(all_model_result, f, indent=2)
    

    print("Pipeline complete. Model score saved to model_results.json")


if __name__ == "__main__":
    main()

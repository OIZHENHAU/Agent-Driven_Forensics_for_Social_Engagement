from src.post_detect.p1_data_cleaning import clean_dataset
from src.post_detect.p2_eda import exploratory_analysis
from src.post_detect.p3_pca import perform_PCA
from src.post_detect.p4_ml_agent import main as train_model

from src.account_detect.p1_fake_account_cleaning import clean_dataset_account
from src.account_detect.p2_fake_account_eda import exploratory_analysis_account
from src.account_detect.p3_fake_account_pca import perform_PCA_account
from src.account_detect.p4_fake_account_detection import main as training_model_account


def main():
    '''Post Detection'''
    # Task 1
    cleaned_df = clean_dataset()
    # Task 2
    exploratory_analysis(cleaned_df)
    # Task 3
    perform_PCA(cleaned_df)
    # Task 4
    train_model(cleaned_df)

    '''Account Detection'''
    # Task 1
    # cleaned_account_df = clean_dataset_account()
    # Task 2
    # exploratory_analysis_account(cleaned_account_df)
    #Task 3
    # perform_PCA_account(cleaned_account_df)
    #Task 4
    # training_model_account(cleaned_account_df)


if __name__ == "__main__":
    main()


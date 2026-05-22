from src.p1_data_cleaning import clean_dataset
from src.p2_eda import exploratory_analysis
from src.p3_pca import perform_PCA
from src.p4_ml_agent import main as train_model


def main():
    # Task 1
    cleaned_df = clean_dataset()
    # Task 2
    exploratory_analysis(cleaned_df)
    # Task 3
    pca, X_pca = perform_PCA(cleaned_df)
    # Task 4
    train_model(X_pca, cleaned_df)


if __name__ == "__main__":
    main()

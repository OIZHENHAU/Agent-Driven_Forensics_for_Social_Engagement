import pandas as pd
import numpy as np
import os
from sklearn.utils import resample

# Get Raw Data from File Path
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw_user_activities.csv")
# The csv file path after data cleaning
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned_data.csv")


# Load Raw CSV file
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df


# Display the dataset information
def dataset_info(df: pd.DataFrame) -> None:
    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Name:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing value:")
    print(df.isnull().sum())


# Remove duplicate values in the dataset.
def remove_duplicates(df):
    duplicates_before = df.duplicated().sum()

    df = df.drop_duplicates()

    duplicates_after = df.duplicated().sum()

    print("\n")
    print("Duplicates Before:", duplicates_before)
    print("Duplicates After:", duplicates_after)

    return df


# Handle missing value by relaceing median for numerical columns, and mode for categorical columns
def handle_missing_values(df):
    print("\nMissing value before handling missing value:")
    print(df.isnull().sum())

    numeric_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(include="object").columns

    # Fill numeric columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())


    # Fill categorical columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing value after handling missing value:")
    print(df.isnull().sum())

    return df


# Remove the outlier of the numerical columns in the datset using IQR.
def remove_outliers(df):
    print("\n")
    print("Dataset Shape Before removing outlier:")
    print(df.shape)
    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        # Skip columns with zero IQR
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


    print("\n")
    print("Dataset Shape After removing outlier:")
    print(df.shape)

    return df

# Resample the dataset so that the number of REAL and FAKE post is the same, to reduce data imbalance
def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    real_data = df[df["is_fake"] == False]
    fake_data = df[df["is_fake"] == True]
    
    print("Real rows before resample: ", len(real_data))
    print("Fake rows before resample: ", len(fake_data))

    real_downsampled = resample(real_data, replace=False, n_samples=len(fake_data), random_state=42)
    balanced_df = pd.concat([real_downsampled, fake_data]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Real rows after resample: ", len(real_downsampled))
    print("Fake rows after resample: ", len(fake_data))
    print("Total balanced rows: ", len(balanced_df))
    
    return balanced_df


# Convert string or boolean columns to numbers
def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    skip_columns = {"activity_id", "user_id", "timestamp", "content"}

    for col in df.columns:
        if col in skip_columns:
            continue

        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col], _ = pd.factorize(df[col].astype(str))

    print("\n")
    print("Column dtypes after encoding:")
    print(df.dtypes)
    return df


# Save the filered dataset into a new dataset and save it into a new path
def save_clean_dataset(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned data saved successfully to the path: {CLEAN_PATH}")


def clean_dataset():
    df = load_dataset()
    dataset_info(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    # df = remove_outliers(df)
    balance_df = resample_data(df)
    encode_df = encode_columns(balance_df)
    save_clean_dataset(encode_df)

    print("Task 1 done.")

    cleaned_df = pd.read_csv(CLEAN_PATH)
    return cleaned_df





import pandas as pd
import numpy as np
import os

# Get Raw Data from File Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(BASE_DIR, "data", "fake_social_media_global_2_0_with_missing.xlsx")

# The csv file path after data cleaning
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

# Load Raw CSV file
def load_dataset() -> pd.DataFrame:
    df = pd.read_excel(RAW_PATH)
    return df

# Print a summary of missing value
def report_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    report  = pd.concat([missing, pct], axis=1, keys=["Missing Count", "Missing %"])
    report  = report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    print(report.to_string())   


# Perform data cleaning
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw social media dataset by handling missing values, 
    correcting data types, and engineering specialized forensic metrics.

    Processing Steps:
    1. Impute missing categorical values with the mode (most frequent value).
    2. Impute missing binary indicators with 0 (assuming missing implies 'False').
    3. Impute missing continuous/numerical data with the column median to avoid outlier skew.
    4. Restore integer data types for discrete count columns (e.g. followers).
    5. Handle zero-division edge cases for ratio calculations.
    6. Compute new forensic features such as engagement ratios.

    Args:
        df (pd.DataFrame): Raw dataframe with missing values.

    Returns:
        pd.DataFrame: A fully cleaned dataframe with no missing values.
    """
    df = df.copy()

    # Step 1: Categorical Imputation
    # If the 'platform' column has a mode (most common value), fill missing cells with it.
    # Otherwise, fallback to 'Unknown' to prevent NaNs.
    if not df["platform"].mode().empty:
        df["platform"] = df["platform"].fillna(df["platform"].mode().iloc[0])
    else:
        df["platform"] = df["platform"].fillna("Unknown")

    # Step 2: Binary Imputation
    # Binary columns represent boolean states (1=Yes, 0=No). 
    # Missing values are safely assumed to be 0 (e.g., if we don't know if they are verified, assume they are not).
    binary_cols = ["has_profile_pic", "verified", "is_fake", "suspicious_links_in_bio"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Step 3: Continuous/Numerical Imputation
    # Use the median instead of the mean to fill missing numbers. 
    # Median is highly robust against extreme outliers (which are common in bot accounts).
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = num_cols.drop(binary_cols, errors='ignore')
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Step 4: Discrete Value Normalization
    # Since we used the median, some counts might have become floats (e.g., 45.5 followers).
    # We round these back to whole numbers and cast to int to make logical sense.
    int_cols = ["bio_length", "followers", "following", "account_age_days", "posts"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    # Step 5 & 6: Feature Engineering & Edge-case Handling
    # To calculate the follower_following_ratio, we must divide by 'following'.
    # If an account follows 0 people, division by zero yields an infinite error.
    # We temporarily replace 0 with 1 in the 'following' column to mathematically stabilize the ratio.
    # Note: replace({0: 1}) is used to avoid Pandas 3.0 deprecation warnings.
    df["following"] = df["following"].replace({0: 1})

    df["follower_following_ratio"] = (df["followers"] / df["following"]).round(4)

    # Calculate Suspicious Engagement Rate:
    # This combines spam and generic comments, divided by total posts. 
    # Again, we replace 0 posts with 1 during division to prevent 'Inf' or 'NaN' errors.
    df["suspicious_engagement_rate"] = (
        (df["spam_comments_rate"] + df["generic_comment_rate"]) /
        df["posts"].replace({0: 1})
    ).round(6)

    return df 

# Check if there is any missing, NaN or null value cell
def validateCell(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "NaN still exists."
    assert (df["followers"] >= 0).all(), "Negative followers found"
    if "is_fake" in df.columns:
        assert df["is_fake"].isin([0, 1]).all(), "is_fake not 0 or 1."
    print("No missing, NaN or null value cell found.")


# Save the filtered dataset into a new dataset and save it into a new path
def save_clean_dataset(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    try:
        df.to_csv(CLEAN_PATH, index=False)
        print(f"\nCleaned data saved successfully to the path: {CLEAN_PATH}")
    except PermissionError:
        print(f"\n[ERROR] Permission denied when trying to save to {CLEAN_PATH}.")
        print("Please ensure the file is NOT open in another program (like Excel) and try again.")


def main():
    print("Loading dataset...")
    df = load_dataset()
    print("Missing values report before cleaning:")
    report_missing(df)
    
    print("\nPerforming data cleaning...")
    df = data_cleaning(df)
    
    validateCell(df)
    save_clean_dataset(df)
    print("Data cleaning has completed.")
    return df


if __name__ == "__main__":
    main()

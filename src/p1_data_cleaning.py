import pandas as pd
import numpy as np
import os

# Get Raw Data from File Path
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                          "fake_social_media_global_2_0_with_missing.xlsx")
# The csv file path after data cleaning
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                          "cleaned_data.csv")

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


# Perform datac cleaning
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill in the categorical part
    df["platform"] = df["platform"].fillna(df["platform"].mode().iloc[0] if not df["platform"].mode().empty else "Unknown")

    # Fill numeric with median 
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))


    df["following"] = df["following"].replace(0, 1)

    df["follower_following_ratio"] = (df["followers"] / df["following"]).round(4)

    df["suspicious_engagement_rate"] = ((df["spam_comments_rate"] + df["generic_comment_rate"])).round(6)

    binary_cols = ["has_profile_pic", "verified", "is_fake", "suspicious_links_in_bio"]
    df[binary_cols] = df[binary_cols].fillna(0).astype(int)

    return df 


# Check if there is any missing, NaN or null value cell
def validateCell(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "NaN still exists."
    assert (df["followers"] >= 0).all(), "Negative followers found"
    assert df["is_fake"].isin([0, 1]).all(), "is_fake not 0 or 1."
    print("No missing, NaN or null value cell found.")


# Save the filered dataset into a new dataset and save it into a new path
def save_clean_dataset(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned data saved successfully to the path: {CLEAN_PATH}")


def main():
    df = load_dataset()
    report_missing(df)
    df = data_cleaning(df)
    validateCell(df)
    save_clean_dataset(df)
    print("Data celaning has completed.")
    return df


if __name__ == "__main__":
    main()



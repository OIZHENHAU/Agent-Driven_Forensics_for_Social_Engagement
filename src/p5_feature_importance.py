import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os, sys, json, io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Get the path for cleaned datasert
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs")
# Makre the output path if does not exist
os.makedirs(OUTPUT_PATH, exist_ok=True)


# Load the cleaned dataset
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    return df

# Get which features are importance so that we can used to train for our agent
def compute_feature_importance(df):
    exclude_columns = ['is_fake', 'username', 'platform']
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    features = [col for col in numeric_columns if col not in exclude_columns]
    
    X = df[features]
    y = df["is_fake"] 
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Get all the list of feature importance
    importance = model.feature_importances_
    
    return pd.DataFrame({ "feature": features, "importance": importance}).sort_values(by="importance", ascending=False)


def plot_feature_importance(importance_df):

    file_path = os.path.join(OUTPUT_PATH, "feature_importance.png")

    plt.figure(figsize=(20, 12))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()

    plt.savefig(file_path)
    plt.close()
    
    print(f"Feature importance plot have successfully saved to: {OUTPUT_PATH}" )
    # plt.show()



df = load_dataset()
importance_feature = compute_feature_importance(df)
plot_feature_importance(importance_feature)


'''def main():
    df = load_dataset()
    importance_feature = compute_feature_importance(df)
    plot_feature_importance(importance_feature)


if __name__ == "__main__":
    main()
'''
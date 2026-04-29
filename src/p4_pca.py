import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Get the path for cleaned datasert
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
# Get the output directory
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs")
# get the PCD dataset path
PCA_PATH = os.path.join(os.path.dirname(__file__), "..", "pca_data.csv")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Columns that need to exclude
exclude_cols = ['is_fake', 'username', 'platform']

# Select only numeric columns
# numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Define features for PCA (exclude target 'is_fake' and non-numeric/ID columns)
# exclude_cols = ['is_fake', 'username', 'platform', 'verified']
# numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# features = [col for col in numeric_cols if col not in exclude_cols]

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    return df


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Get the path for cleaned datasert
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")

# Load the cleaned dataset
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    return df


import io
import os
import sys
import math
import threading
import webbrowser

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from backend.src.p1_data_cleaning import (
    load_dataset, dataset_info, remove_duplicates, handle_missing_values, 
    remove_outliers, save_clean_dataset
    )

from backend.src.p2_eda import (
    plot_histogram, plot_boxplots, correlation_heatmap, validate_log_normal
)

from backend.src.p3_pca import (
    select_features, normalize_features, apply_pca, plot_explained_variance,
    plot_pca_scatter, perform_PCA
)

from backend.src.p4_ml_agent import (
    engineer_features, scale_features, train_isolation_forest, train_lof,
    find_best_threshold, plot_anomaly_results, plot_confusion_matrix, evaluate_model
)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.get('/')
def index():
    return app.send_static_file('index.html')



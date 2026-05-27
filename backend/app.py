import io
import os
import sys
import math
import threading
import webbrowser

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

from src.post_detect.p1_data_cleaning import (load_dataset, dataset_info, remove_duplicates, 
                                              handle_missing_values, remove_outliers, save_clean_dataset)

from src.post_detect.p2_eda import (plot_histogram, plot_boxplots, correlation_heatmap, validate_log_normal)

from src.post_detect.p3_pca import (select_features, normalize_features, apply_pca, 
                                    plot_explained_variance, plot_pca_scatter, perform_PCA)

from src.post_detect.p4_ml_agent import (engineer_features, scale_features, train_isolation_forest, 
                                         train_lof, plot_anomaly_results, plot_confusion_matrix, evaluate_model)

from src.account_detect.p4_fake_account_detection import predict_single_account
from src.post_detect.p5_single_post_detection import predict_single_post
from utils.gemini_service import analyze_account_with_gemini, analyze_post_with_gemini
from pipeline import main as run_pipeline

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.get('/')
def index():
    return app.send_static_file('index.html')


@app.post('/api/analyze-account')
def analyze_account_endpoint():
    data = request.get_json(force=True)
    try:
        result = predict_single_account(data)
        try:
            result['gemini_explanation'] = analyze_account_with_gemini(data, result)

        except Exception as gemini_err:
            result['gemini_explanation'] = f"(AI analysis unavailable: {gemini_err})"

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/api/analyze-post')
def analyze_post_endpoint():
    data = request.get_json(force=True)
    try:
        result = predict_single_post(data)
        try:
            result['gemini_explanation'] = analyze_post_with_gemini(data, result)
        except Exception as gemini_err:
            result['gemini_explanation'] = f"(AI analysis unavailable: {gemini_err})"
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        threading.Thread(target=run_pipeline, daemon=True).start()
        
    app.run(debug=True, port=5000)


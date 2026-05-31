import io
import os
import sys
import math
import json
import threading
import webbrowser
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, jsonify, request, send_from_directory

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

from src.post_detect.p1_data_cleaning import (load_dataset, dataset_info, remove_duplicates, 
                                              handle_missing_values, remove_outliers, save_clean_dataset)

from src.post_detect.p2_eda import (plot_histogram, plot_boxplots, correlation_heatmap, validate_log_normal)

from src.post_detect.p3_pca import (select_features, normalize_features, apply_pca, 
                                    plot_explained_variance, plot_pca_scatter, perform_PCA)

from src.post_detect.p4_ml_agent import (engineer_features, normalize_features, train_isolation_forest, 
                                         train_lof, plot_anomaly_results, plot_confusion_matrix, evaluate_model)

from src.account_detect.p4_fake_account_detection import predict_single_account, predict_batch_accounts
from src.post_detect.p5_single_post_detection import predict_single_post, predict_batch_posts
from utils.gemini_service import analyze_account_with_gemini, analyze_post_with_gemini, generate_model_report_with_gemini
from pipeline import main as run_pipeline


MODEL_RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model_result.json")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


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


@app.post('/api/analyze-csv')
def analyze_csv_endpoint():
    csv_type = request.form.get('type', 'account')
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        filename = file.filename or ''
        if filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        rows = df.to_dict(orient='records')
        if not rows:
            return jsonify({'error': 'CSV file is empty'}), 400
        if csv_type == 'post':
            results = predict_batch_posts(rows)
        else:
            results = predict_batch_accounts(rows)
        authentic = sum(1 for r in results if r['verdict'] == 'Authentic')
        return jsonify({
            'type': csv_type,
            'total': len(results),
            'authentic': authentic,
            'suspicious': len(results) - authentic,
            'results': results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/api/analyze-csv-row')
def analyze_csv_row_endpoint():
    data = request.get_json(force=True)
    csv_type = data.get('type', 'account')
    row_data = data.get('row_data', {})
    scores = data.get('scores', {})
    try:
        if csv_type == 'post':
            explanation = analyze_post_with_gemini(row_data, scores)
        else:
            explanation = analyze_account_with_gemini(row_data, scores)
        return jsonify({'gemini_explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.get('/api/get_model_result')
def get_pipeline_model_results():
    if not os.path.exists(MODEL_RESULT_PATH):
        return jsonify({'error': 'Pipeline not yet finish running. '})
    
    with open(MODEL_RESULT_PATH) as result:
        return jsonify(json.load(result))


# API to generate the summary AI report
@app.post('/api/generate-model-report')
def generate_model_report():
    data = request.get_json(force=True)
    model_result = data.get('modelResults', {})

    try:
        report = generate_model_report_with_gemini(model_result)
        return jsonify({'report': report})
    
    except Exception as error:
        return jsonify({'error': str(error)}), 500


@app.get('/api/output-images/<category>')
def list_output_images(category):
    if category not in ('post', 'account'):
        return jsonify({'error': 'Invalid category'}), 400
    
    image_folder = os.path.join(OUTPUTS_DIR, category)
    if not os.path.exists(image_folder):
        return jsonify({'images': []})
    
    images = sorted(f for f in os.listdir(image_folder) if f.lower().endswith('.png'))
    return jsonify({'images': images})


@app.get('/outputs/<category>/<filename>')
def serve_output_image(category, filename):
    if category not in ('post', 'account'):
        return jsonify({'error': 'Invalid category'}), 400
    return send_from_directory(os.path.join(OUTPUTS_DIR, category), filename)


if __name__ == '__main__':
    threading.Thread(target=run_pipeline, daemon=True).start()
    app.run(debug=True, use_reloader=False, port=5000)


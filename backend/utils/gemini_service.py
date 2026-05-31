import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

MODEL = "google/gemini-2.0-flash-001"

CLEAN_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_account_data.csv")
CLEAN_POST_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_data.csv")


ACCOUNT_SYSTEM_PROMPT = """\
You are an investigative social media forensics agent analyzing a social media account for authenticity.

The Isolation Forest and LOF authenticity scores and the feature anomaly analysis have already been computed and are provided to you in the user message.

Your job: Write a forensic report using ONLY the format below. Do NOT output any JSON, code blocks, or raw data.

Required output format (plain text only):
Verdict: [Fake | Suspicious | Authentic]
Confidence: [Low | Medium | High]
Ensemble Score: [use the provided ensemble score] / 100

Red Flags:
- [one bullet per anomalous feature from the anomaly analysis; write "None detected" if no anomalies]

Analysis:
[2-3 sentences citing the provided scores and the most anomalous features]

Be direct and factual. Output plain text only — no markdown code blocks, no JSON.\
"""

POST_SYSTEM_PROMPT = """\
You are an investigative social media forensics agent analyzing a social media post for authenticity.

The Isolation Forest and LOF authenticity scores have already been computed and are provided to you.

Your job: review the post content, engagement metrics, lexical diversity, and ML scores, then write a forensic report.

Required output format:
Verdict: [Fake | Suspicious | Authentic]
Confidence: [Low | Medium | High]
Ensemble Score: [use the provided ensemble score] / 100
Lexical Diversity (TTR): [provided value] — [brief interpretation]

Red Flags:
- [one bullet per anomalous features and write "None detected" if clean]

Analysis:
- [2-3 sentences citing the provided scores, lexical diversity, and engagement patterns]

Be direct and factual.\
"""


# Analyse the post authentication score using GEMINI AI
def analyze_post_with_gemini(post_data: dict, ml_scores: dict) -> str:
    if_score = ml_scores.get("if_score", "N/A")
    lof_score = ml_scores.get("lof_score", "N/A")
    ensemble = ml_scores.get("ensemble_score", "N/A")
    verdict = ml_scores.get("verdict", "Unknown")
    ld = ml_scores.get("lexical_diversity", "N/A")

    user_content = (
        f"Analyze this Instagram post for authenticity.\n\n"
        f"Post data: {post_data}\n\n"
        f"Pre-computed ML scores (use these exact values in your report):\n"
        f"- Isolation Forest authenticity score: {if_score}/100\n"
        f"- LOF authenticity score: {lof_score}/100\n"
        f"- Ensemble score: {ensemble}/100\n"
        f" - Verdict: {verdict}\n"
        f"Feature anomaly analysis (percentile-based, vs real-post baseline):\n"
        f"Write your forensic report now."
    )

    messages = [{"role": "system", "content": POST_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content or "(No forensic analysis generated)"


# Analyse the account authentication score using GEMJNIN AI
def analyze_account_with_gemini(account_data: dict, ml_scores: dict) -> str:
    if_score = ml_scores.get("if_score", "N/A")
    lof_score = ml_scores.get("lof_score", "N/A")
    ensemble = ml_scores.get("ensemble_score", "N/A")
    verdict = ml_scores.get("verdict", "Unknown")

    user_content = (
        f"Analyze this Instagram account for authenticity.\n\n"
        f"Account data: {account_data}\n\n"
        f"Pre-computed ML scores (use these exact values in your report):\n"
        f"- Isolation Forest authenticity score: {if_score}/100\n"
        f"- LOF authenticity score: {lof_score}/100\n"
        f"- Ensemble score: {ensemble}/100\n"
        f" - Verdict: {verdict}\n"
        f"Feature anomaly analysis (percentile-based, vs real-account baseline):\n"
        f"Write your forensic report now."
    )

    messages = [{"role": "system", "content": ACCOUNT_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content or "(No forensic analysis generated)"


# Generate the precision-recall repost explanation of the two models
def generate_model_report_with_gemini(metrics: dict) -> str:
    post = metrics.get("post_detection", {})
    acct = metrics.get("account_detection", {})

    def fmt(m):
        return (f"Accuracy={m.get('accuracy', 0):.4f}, Precision={m.get('precision', 0):.4f}, "
                f"Recall={m.get('recall', 0):.4f}, F1={m.get('f1', 0):.4f}")

    user_content = (
        "You are an investigative AI forensics agent. Analyse the performance of two "
        "unsupervised anomaly detection models — Isolation Forest (IF) and Local Outlier "
        "Factor (LOF) — applied to both post and account authenticity detection.\n\n"
        "POST DETECTION RESULTS:\n"
        f"Isolation Forest : {fmt(post.get('isolation_forest', {}))}\n"
        f"Local Outlier Factor: {fmt(post.get('lof', {}))}\n\n"
        "ACCOUNT DETECTION RESULTS:\n"
        f"Isolation Forest : {fmt(acct.get('isolation_forest', {}))}\n"
        f"Local Outlier Factor: {fmt(acct.get('lof', {}))}\n\n"
        "Write a rigorous forensic report covering exactly these four sections:\n\n"
        "**1. Model Performance Summary**\n"
        "Interpret each model's metrics for both post and account detection.\n\n"
        "**2. Precision-Recall Trade-off Analysis**\n"
        "Explain why high recall often comes at the cost of precision in unsupervised "
        "outlier detection, referencing the actual numbers above. Discuss how the "
        "contamination hyperparameter drives this trade-off.\n\n"
        "**3. Business Implications**\n"
        "What do these metrics mean for a B2B client auditing influencer accounts and posts?\n\n"
        "**4. Recommendations**\n"
        "How should the contamination threshold be tuned depending on whether the client "
        "prioritises recall (catch all fakes) or precision (minimise false accusations)?\n\n"
        "Be analytical, cite the specific numbers, and explain the inherent limitations "
        "of unsupervised detection without labelled data."
    )

    messages = [{"role": "user", "content": user_content}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content or "(No report generated)"

import os
import json
import numpy as np
import pandas as pd
import google.generativeai as genai
# from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
# MODEL = "openrouter/free"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel("gemini-2.5-flash")


CLEAN_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_account_data.csv")
CLEAN_POST_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_data.csv")


ACCOUNT_SYSTEM_PROMPT = """\
You are an investigative social media forensics agent analyzing a social media account for authenticity.
The Isolation Forest and LOF authenticity scores have already been computed and are provided to you.
Your job: review the account content, profile signals, and ML scores, then identify suspicious patterns and write a forensic report.


Required output format:
Verdict: [Fake | Suspicious | Authentic]
Confidence: [Low | Medium | High]
Ensemble Score: [use the provided ensemble score] / 100

Red Flags:
- [one bullet per anomalous features from the anomaly analysis; write "None detected" if clean]

Analysis:
- [2-3 sentences citing the provided scores and the most suspicious account features]

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
- [one bullet per anomalous features from the anomaly analysis; write "None detected" if clean]

Analysis:
- [2-3 sentences citing the provided scores, lexical diversity, and engagement patterns]

Be direct and factual. Output plain text only — no markdown code blocks, no JSON.\
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

    # messages = [{"role": "system", "content": POST_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    # response = client.chat.completions.create(model=MODEL, messages=messages)
    # return response.choices[0].message.content or "(No forensic analysis generated)"\

    final_prompt = f"""
        {POST_SYSTEM_PROMPT}

        {user_content}
    """
    try:
        response = MODEL.generate_content(final_prompt)
        return response.text 
    
    except Exception as error:
        return f"(Fail to generated the post analysing due to: {str(error)})"



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
        f"- Verdict: {verdict}\n\n"
        f"Review the account data above and identify any suspicious signals. Write your forensic report now."
    )

    # messages = [{"role": "system", "content": ACCOUNT_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    # response = client.chat.completions.create(model=MODEL, messages=messages)
    # return response.choices[0].message.content or "(No forensic analysis generated)"

    final_prompt = f"""
        {ACCOUNT_SYSTEM_PROMPT}

        {user_content}
    """

    try:
        response = MODEL.generate_content(final_prompt)
        return response.text 
    
    except Exception as error:
        return f"(Fail to generated the account analysing due to: {str(error)})"



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
    )

    # messages = [{"role": "user", "content": user_content}]
    # response = client.chat.completions.create(model=MODEL, messages=messages)
    # return response.choices[0].message.content or "(No report generated)"

    try:
        response = MODEL.generate_content(user_content)
        return response.text
    
    except Exception as error:
        return f"(Fail to generate the summary of the precision-recall report due to: {str(error)})"

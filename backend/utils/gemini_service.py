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
training_df = None

def get_training_data() -> pd.DataFrame:
    global training_df
    if training_df is None:
        training_df = pd.read_csv(CLEAN_ACCOUNT_PATH)
    return training_df


def account_data_summary(account_data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "profile pic": int(account_data.get("profile_pic", 0)),
        "nums/length username": float(account_data.get("nums_length_username", 0)),
        "fullname words": int(account_data.get("fullname_words", 1)),
        "nums/length fullname": float(account_data.get("nums_length_fullname", 0)),
        "name==username": int(account_data.get("name_equals_username", 0)),
        "description length": int(account_data.get("description_length", 0)),
        "external URL": int(account_data.get("external_url", 0)),
        "private": int(account_data.get("private", 0)),
        "#posts": int(account_data.get("posts", 0)),
        "#followers": int(account_data.get("followers", 0)),
        "#follows": int(account_data.get("follows", 0)),
    }])


def engineer_account_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1
    features = pd.DataFrame(index=df.index)
    for column in df.select_dtypes(include="number").columns:
        if column != "fake":
            features[column] = df[column]

    features["follower_following_ratio"] = df["#followers"] / (df["#follows"] + eps)
    features["posts_per_follower"] = df["#posts"] / (df["#followers"] + eps)
    features["following_per_follower"] = df["#follows"] / (df["#followers"] + eps)
    features["posts_per_following"] = df["#posts"] / (df["#follows"] + eps)
    features["has_description"] = (df["description length"] > 0).astype(int)
    features["log_posts"] = np.log1p(df["#posts"])
    features["log_followers"] = np.log1p(df["#followers"])
    features["log_follows"] = np.log1p(df["#follows"])

    return features


def get_score(val: float, all_vals: np.ndarray) -> int:
    low_score, high_score = all_vals.min(), all_vals.max()
    if high_score == low_score:
        return 50
    return int(np.clip((val - low_score) / (high_score - low_score) * 100, 0, 100))


def isolation_forest_tool(account_data: dict) -> dict:
    df = get_training_data()
    single = account_data_summary(account_data)
    combo = pd.concat([df.drop(columns=["fake"]), single], ignore_index=True)

    X = StandardScaler().fit_transform(engineer_account_features(combo))
    X_real = X[:len(df)][df["fake"].values == 0]
    cont = float(np.clip(df["fake"].mean(), 0.05, 0.45))

    model = IsolationForest(n_estimators=300, contamination=cont, max_features=1.0, random_state=42, n_jobs=-1)
    model.fit(X_real)
    train_scores = model.decision_function(X[:len(df)])
    new_score = float(model.decision_function(X[-1:])[0])
    score = get_score(new_score, train_scores)

    return {"isolation_forest_authenticity_score": score, "verdict": "Authentic" if score >= 50 else "Suspicious",
            "note": "0 = fully fake, 100 = fully authentic. The isolation forest model detects multivariate outliers."}


def lof_tool(account_data: dict) -> dict:
    df = get_training_data()
    single = account_data_summary(account_data)
    combo = pd.concat([df.drop(columns=["fake"]), single], ignore_index=True)
    X = StandardScaler().fit_transform(engineer_account_features(combo))
    X_real = X[:len(df)][df["fake"].values == 0]
    cont = float(np.clip(df["fake"].mean(), 0.05, 0.45))

    model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=cont, n_jobs=-1)
    model.fit(X_real)
    train_scores = model.decision_function(X[:len(df)])
    new_score = float(model.decision_function(X[-1:])[0])
    score = get_score(new_score, train_scores)

    return {"lof_authenticity_score": score, "verdict": "Authentic" if score >= 50 else "Suspicious",
            "note": "0 = fake, 100 = authentic. LOF measures local density vs real-account neighbourhoods."}


def get_feature_anomalies_tool(account_data: dict) -> dict:
    df = get_training_data()
    real = df[df["fake"] == 0]

    mapping = {"profile_pic": "profile pic", "nums_length_username": "nums/length username",
               "fullname_words": "fullname words", "nums_length_fullname": "nums/length fullname",
               "name_equals_username": "name==username", "description_length": "description length",
               "external_url": "external URL", "private": "private", "posts": "#posts",
               "followers": "#followers", "follows": "#follows"}

    flags_list = []
    for key, column in mapping.items():
        value = account_data.get(key)
        if value is None or column not in real.columns:
            continue

        mean, standard_deviation = real[column].mean(), real[column].std()
        if standard_deviation < 1e-6:
            continue

        z_score = abs((float(value) - mean) / standard_deviation)
        if z_score > 1.5:
            flags_list.append({"feature": column, "account_value": float(value),
                                "typical_range": f"{mean - standard_deviation:.2f} to {mean + standard_deviation:.2f}",
                                "z_score": round(z_score, 2), "severity": "High" if z_score > 3.0 else "Medium"})

    flags_list.sort(key=lambda x: x["z_score"], reverse=True)
    return {"anomalous_features": flags_list[:5], "total_anomalies": len(flags_list),
            "summary": f"Found {len(flags_list)} features deviating significantly (z-score > 1.5) from the real-account baseline."}


TOOL_DISPATCH = {"isolation_forest_tool": isolation_forest_tool, "lof_tool": lof_tool, "get_feature_anomalies_tool": get_feature_anomalies_tool,}

ACCOUNT_PARAMS = {
    "type": "object",
    "properties": {
        "profile_pic": {"type": "integer", "description": "1 = has profile picture, 0 = no picture"},
        "nums_length_username": {"type": "number", "description": "Fraction of username chars that are digits (0-1)"},
        "fullname_words": {"type": "integer", "description": "Number of words in the full name"},
        "nums_length_fullname": {"type": "number", "description": "Fraction of full-name chars that are digits (0-1)"},
        "name_equals_username": {"type": "integer", "description": "1 = full name identical to username"},
        "description_length": {"type": "integer", "description": "Character count of bio/description"},
        "external_url": {"type": "integer", "description": "1 = bio contains an external URL"},
        "private": {"type": "integer", "description": "1 = private account"},
        "posts": {"type": "integer", "description": "Total number of posts"},
        "followers": {"type": "integer", "description": "Follower count"},
        "follows": {"type": "integer", "description": "Following count"},
    },
    "required": ["followers", "follows", "posts"],
}

TOOLS = [
    {"type": "function", "function": {"name": "isolation_forest_tool",
        "description": "Run the Isolation Forest unsupervised anomaly detection model. Returns an authenticity score 0-100.",
        "parameters": ACCOUNT_PARAMS}},
    {"type": "function", "function": {"name": "lof_tool",
        "description": "Run the Local Outlier Factor density-based anomaly detection model. Returns an authenticity score 0-100.",
        "parameters": ACCOUNT_PARAMS}},
    {"type": "function", "function": {"name": "get_feature_anomalies_tool",
        "description": "Compare account features against the statistical baseline of verified real accounts. Returns top anomalous features ranked by z-score.",
        "parameters": ACCOUNT_PARAMS}},
]

SYSTEM_PROMPT = """\
You are an investigative social media forensics agent.

The Isolation Forest and LOF authenticity scores have already been computed and are provided to you — use those exact numbers in your report.

Your job:
1. Call get_feature_anomalies_tool with the account features to identify which specific features are anomalous.
2. Write your final forensic report using the exact format below. Use the ML scores provided in the user message.

Required output format:
Verdict: [Fake | Suspicious | Authentic]
Confidence: [Low | Medium | High]
Ensemble Score: [use the provided ensemble score] / 100

Red Flags:
- [one bullet per anomalous feature from get_feature_anomalies_tool; write "None detected" if clean]

Analysis:
- [2-3 sentences citing the provided scores and the most anomalous features found by the tool]

Be direct and factual.\
"""

ANOMALY_TOOLS = [
    {"type": "function", "function": {"name": "get_feature_anomalies_tool",
        "description": "Compare account features against the statistical baseline of verified real accounts. Returns top anomalous features ranked by z-score.",
        "parameters": ACCOUNT_PARAMS}},
]


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
        f"- Ensemble score: {ensemble}/100  →  Verdict: {verdict}\n\n"
        f"Call get_feature_anomalies_tool, then write your forensic report."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    last_content = None

    for _ in range(6):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=ANOMALY_TOOLS, tool_choice="auto")
        message = response.choices[0].message

        if message.content:
            last_content = message.content

        msg_dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            msg_dict["tool_calls"] = [{"id": tc.id, "type": "function", 
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

        if not message.tool_calls:
            return message.content or last_content or "(No forensic analysis generated)"

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            fn = TOOL_DISPATCH.get(tool_call.function.name)
            result = fn(args) if fn else {"error": f"Unknown tool: {tool_call.function.name}"}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

        if "get_feature_anomalies_tool" in {tc.function.name for tc in message.tool_calls}:
            messages.append({"role": "user", "content": "Now write your forensic report using the provided ML scores and the anomaly results above."})

    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content or last_content or "(No forensic analysis generated)"

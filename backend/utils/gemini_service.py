import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings("ignore", message=".*google.generativeai.*", category=FutureWarning)
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
        "nums/length username" : float(account_data.get("nums_length_username", 0)),
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


# Train the isolation forest model on real account data
def isolation_forest_tool(account_data: dict) -> dict:
    df = get_training_data()
    single = account_data_summary(account_data)
    combo  = pd.concat([df.drop(columns=["fake"]), single], ignore_index=True)

    X = StandardScaler().fit_transform(engineer_account_features(combo))
    X_real = X[:len(df)][df["fake"].values == 0]
    cont = float(np.clip(df["fake"].mean(), 0.05, 0.45))

    model = IsolationForest(n_estimators=300, contamination=cont, max_features=1.0, random_state=42, n_jobs=-1)
    model.fit(X_real)
    scores = model.decision_function(X)
    score  = get_score(float(scores[-1]), scores)

    return {"isolation_forest_authenticity_score": score, "verdict": "Authentic" if score >= 50 else "Suspicious",
        "note": "0 = fully fake, 100 = fully authentic. The isolation forest model detects multivariate outliers."}


# Train the LOF model on real-account data.
def lof_tool(account_data: dict) -> dict:
    df = get_training_data()
    single = account_data_summary(account_data)
    combo = pd.concat([df.drop(columns=["fake"]), single], ignore_index=True)
    X = StandardScaler().fit_transform(engineer_account_features(combo))
    X_real = X[:len(df)][df["fake"].values == 0]

    model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean", contamination=0.5, n_jobs=-1)
    model.fit(X_real)
    scores = model.decision_function(X)
    score  = get_score(float(scores[-1]), scores)

    return {
        "lof_authenticity_score": score, "verdict": "Authentic" if score >= 50 else "Suspicious",
        "note": "0 = fake, 100 = authentic. LOF measures local density vs real-account neighbourhoods."}


# Compare each account feature against the mean and starndard deviation of verified real accounts. 
# Returns features with z-score > 1.5.
def get_feature_anomalies_tool(account_data: dict) -> dict:
    df = get_training_data()
    real = df[df["fake"] == 0]

    mapping = {"profile_pic": "profile pic", "nums_length_username": "nums/length username", "fullname_words": "fullname words",
        "nums_length_fullname": "nums/length fullname", "name_equals_username": "name==username",
        "description_length": "description length", "external_url": "external URL", "private": "private", "posts": "#posts",
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
                "z_score": round(z_score, 2), "severity": "High" if z_score > 3.0 else "Medium",})

    flags_list.sort(key=lambda x: x["z_score"], reverse=True)
    return {"anomalous_features": flags_list[:5], "total_anomalies": len(flags_list),
        "summary": (f"Found {len(flags_list)} features deviating significantly " 
                    f"(z-score > 1.5) from the real-account baseline.")}


# Gemini tool schema declaratiOn
ACCOUNT_VALUE = {
    "type": "object",
    "properties": {
        "profile_pic": {"type": "integer", "description": "1 = has profile picture, 0 = no picture"},
        "nums_length_username": {"type": "number",  "description": "Fraction of username chars that are digits (0-1)"},
        "fullname_words": {"type": "integer", "description": "Number of words in the full name"},
        "nums_length_fullname": {"type": "number",  "description": "Fraction of full-name chars that are digits (0-1)"},
        "name_equals_username": {"type": "integer", "description": "1 = full name identical to username"},
        "description_length": {"type": "integer", "description": "Character count of bio/description"},
        "external_url": {"type": "integer", "description": "1 = bio contains an external URL"},
        "private": {"type": "integer", "description": "1 = private account"},
        "posts": {"type": "integer", "description": "Total number of posts"},
        "followers": {"type": "integer", "description": "Follower count"},
        "follows": {"type": "integer", "description": "Following count"},
    },
    "required": ["followers", "follows", "posts"]
}

TRAINING_TOOLS = [{
    "function_declarations": [
        {
            "name": "isolation_forest_tool",
            "description": ("Run the Isolation Forest unsupervised anomaly detection model on the account. "
            "Trained exclusively on verified real accounts. Returns an authenticity score 0 to 100 (100 = most authentic)."),
            "parameters": ACCOUNT_VALUE,
        },
        {
            "name": "lof_tool",
            "description": ("Run the Local Outlier Factor density-based anomaly detection model. Measures whether the account "
            "sits in a low-density region relative to real-account neighbourhoods. Returns an authenticity score "
            "between 0 to 100."),
            "parameters": ACCOUNT_VALUE,
        },
        {
            "name": "get_feature_anomalies_tool",
            "description": (
                "Compare each account feature against the statistical baseline "
                "mean of verified real accounts. Returns the top anomalous "
                "features ranked by z-score and use these as factual red flags."
            ),
            "parameters": ACCOUNT_VALUE,
        },
    ]
}]

SYSTEM_PROMPT = """\
You are an investigative social media forensics agent with access to machine-learning tools.

Your task: determine whether an Instagram account is authentic or fake.

Mandatory protocol — you must call all three tools before writing your report:
1. Call run_isolation_forest with the account features.
2. Call run_lof with the account features.
3. Call get_feature_anomalies with the account features.
4. Compute the ensemble score: round(0.6 * IF_score + 0.4 * LOF_score).
5. Write your final forensic report using the exact format below.

Required output format (preserve the bold markers):
**Verdict**: [Fake | Suspicious | Authentic]
**Confidence**: [Low | Medium | High]
**Ensemble Score**: [computed value] / 100

**Red Flags**:
- [one bullet per anomalous feature from get_feature_anomalies; write "None detected" if clean]

**Analysis**: [2-3 sentences citing the specific tool scores and the most anomalous metrics]

Be direct and factual. Highlight only what the tool outputs reveal.\
"""



def execute_tool(name: str, args: dict) -> dict:
    dispatch = {"run_isolation_forest": isolation_forest_tool, "run_lof": lof_tool,
                "get_feature_anomalies": get_feature_anomalies_tool}
    fn = dispatch.get(name)

    return fn(args) if fn else {"error": f"Unknown tool: {name}"}


# Gemini tool-calling agent that autonomously runs 2 models, and feature-anomaly 
# tools, then produces a AI structure report.
def analyze_account_with_gemini(account_data: dict) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction = SYSTEM_PROMPT, tools= TRAINING_TOOLS)
    chat = model.start_chat()
    response = chat.send_message(f"Analyze this Instagram account for authenticity using your forensic tools:\n{account_data}")

    for _ in range(8):
        calls = [p.function_call for p in response.parts if p.function_call.name]

        if not calls:
            break

        tool_parts = [genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name,
                            response={"result": execute_tool(fn.name, dict(fn.args))},)) for fn in calls]
        response = chat.send_message(tool_parts)

    for part in response.parts:
        if part.text:
            return part.text

    return "(No forensic analysis generated)"

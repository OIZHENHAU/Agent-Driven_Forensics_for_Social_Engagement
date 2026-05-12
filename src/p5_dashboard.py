import os
from flask import Flask, render_template, send_from_directory, request, jsonify
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "template"),
    static_folder=os.path.join(BASE_DIR, "web", "static")
)

@app.route("/")
def dashboard1():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned_data.csv"))

    total = len(df)
    real = (df["is_fake"] == 0).sum()
    fake = (df["is_fake"] == 1).sum()

    return render_template(
        "dashboard1.html",
        total=total,
        real=real,
        fake=fake
    )

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(os.path.join(BASE_DIR, "outputs"), filename)
if __name__ == "__main__":
    app.run(debug=True)
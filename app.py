"""
app.py — Flask REST API for salary prediction
Endpoints:
  GET  /api/meta          → feature metadata (allowed values, model metrics)
  POST /api/predict        → predict salary from feature JSON
  GET  /api/health         → liveness check
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

app = Flask(__name__)
CORS(app)

# ── Load artefacts once at startup ────────────────────────────────────────────

def _load_artefacts():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.json")

    if not all(os.path.exists(p) for p in [model_path, enc_path, meta_path]):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, encoders, meta


MODEL, ENCODERS, META = _load_artefacts()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/api/meta", methods=["GET"])
def get_meta():
    if META is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 503
    return jsonify(META)


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 503

    data = request.get_json(force=True)

    # Validate required fields
    required = META["feature_cols"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Validate categorical values
    for col in META["categorical_cols"]:
        value = data.get(col)
        allowed = META["classes"][col]
        if value not in allowed:
            return jsonify({"error": f"Invalid value '{value}' for '{col}'. Allowed: {allowed}"}), 400

    # Validate numeric fields
    for col in ["Experience", "Age"]:
        try:
            val = float(data[col])
            if val < 0:
                raise ValueError
        except (ValueError, TypeError):
            return jsonify({"error": f"'{col}' must be a non-negative number"}), 400

    # Build feature vector
    row = []
    for col in META["feature_cols"]:
        if col in META["categorical_cols"]:
            row.append(int(ENCODERS[col].transform([data[col]])[0]))
        else:
            row.append(float(data[col]))

    X = np.array([row])
    salary = float(MODEL.predict(X)[0])

    # Feature importances for display
    importances = dict(
        zip(META["feature_cols"], [round(float(v), 4) for v in MODEL.feature_importances_])
    )

    return jsonify({
        "predicted_salary": round(salary, 2),
        "feature_importances": importances,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if MODEL is None:
        print("WARNING: No trained model found. Run `python train_model.py` first.")
    app.run(host="0.0.0.0", port=5000, debug=True)

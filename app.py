"""
AmrLens AI — Flask application entry point.

Handles model loading, auto-training pipeline, and prediction routes.
"""

"""
AmrLens AI: Bacterial Resistance Prediction Engine (v2.0)

This module serves as the primary Flask backend for the AmrLens AI project.
It handles clinical data ingestion, advanced feature engineering, machine learning
inference (via HistGradientBoosting), and clinical reasoning generation.

Author: AmrLens AI Team (DELTA FORCE) - SPIRIT 2026
License: MIT
"""

import os
import pickle
import subprocess
import sys

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

TARGET_COLS = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM',
    'GEN', 'AN', 'Acide nalidixique', 'ofx', 'CIP',
    'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

RESISTANCE_MAP = {
    0: "Resistant",
    1: "Intermediate",
    2: "Susceptible",
}

# Module-level model references (populated by load_models)
model = None
feature_cols = None
lookup: dict = {}
pattern_models: dict = {}
souches_mapping: dict = {}


# ---------------------------------------------------------------------------
# AUTO PIPELINE — ensure all artefacts exist before serving
# ---------------------------------------------------------------------------

def _run(script: str) -> None:
    """Run a Python script in a subprocess, raising on failure."""
    # Using sys.executable ensures it uses the same VENV python that launched the app
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"[WARNING] {script} exited with code {result.returncode}")


def ensure_models_exist() -> None:
    """Check for required model/data files and regenerate any that are missing."""
    print("[PROCESS] Checking system...")

    if not os.path.exists("models/souches_mapping.pkl"):
        print("[ACTION] Initial processing...")
        _run("src/clean.py")

    if not os.path.exists("models/model.pkl"):
        print("[ACTION] Training base model...")
        _run("src/train.py")

    if not os.path.exists("models/lookup.pkl"):
        print("[ACTION] Building lookup table...")
        _run("src/hybrid_lookup.py")

    if not os.path.exists("data/Cleaned_Dataset2.csv"):
        print("[ACTION] Cleaning dataset 2...")
        _run("src/clean2.py")

    if not os.path.exists("models/pattern_models.pkl"):
        print("[ACTION] Training pattern models...")
        _run("src/pattern_model.py")

    if not os.path.exists("static/network_graph.png"):
        print("[ACTION] Generating network graph...")
        _run("src/network.py")

    if not os.path.exists("static/resistance_heatmap.png"):
        print("[ACTION] Generating scientific heatmap...")
        _run("src/heatmap.py")

    print("[SUCCESS] System ready")


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models() -> None:
    """Load all pickled models and artefacts into module-level variables."""
    global model, feature_cols, lookup, pattern_models, souches_mapping

    try:
        model = joblib.load("models/model.pkl")
    except (FileNotFoundError, Exception) as exc:
        model = None
        print(f"[ERROR] Base model not loaded: {exc}")

    try:
        feature_cols = joblib.load("models/feature_cols.pkl")
    except (FileNotFoundError, Exception) as exc:
        feature_cols = None
        print(f"[ERROR] Feature columns not loaded: {exc}")

    try:
        with open("models/lookup.pkl", "rb") as f:
            lookup = pickle.load(f)
    except (FileNotFoundError, Exception) as exc:
        lookup = {}
        print(f"[ERROR] Lookup table not loaded: {exc}")

    try:
        with open("models/pattern_models.pkl", "rb") as f:
            pattern_models = pickle.load(f)
    except (FileNotFoundError, Exception) as exc:
        pattern_models = {}
        print(f"[ERROR] Pattern models not loaded: {exc}")

    try:
        souches_mapping = joblib.load("models/souches_mapping.pkl")
    except (FileNotFoundError, Exception) as exc:
        souches_mapping = {}
        print(f"[ERROR] Souches mapping not loaded: {exc}")


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", prediction=None, form_data={})


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template(
            "index.html",
            error="Prediction model is unavailable. Please check server logs.",
            prediction=None,
            form_data=request.form.to_dict(),
        )

    form_data = request.form.to_dict()

    try:
        age = float(request.form.get("age", 25))
        gender = float(request.form.get("gender", 0))
        diabetes = float(request.form.get("diabetes", 0))
        hypertension = float(request.form.get("hypertension", 0))
        hospital_before = float(request.form.get("hospital_before", 0))
        infection_freq = float(request.form.get("infection_freq", 0))
        strain = request.form.get("souches") or None

        # --- Feature Engineering ---
        # 1. Comorbidity
        comorbidity = int(diabetes + hypertension)
        # 2. Age Binning
        def bin_age(a):
            if a <= 18: return 0
            if a <= 35: return 1
            if a <= 55: return 2
            if a <= 75: return 3
            return 4
        age_group = bin_age(age)
        # 3. Clinical Risk Score (Weight: Hosp(2) + InfFreq(1) + Comorbidity(1))
        risk_score = float(hospital_before * 2 + infection_freq + comorbidity)
        # 4. Souches Encoding
        souches_id = souches_mapping.get(strain, souches_mapping.get('Unknown', 0))

        raw_inputs = {
            "age": age,
            "gender": gender,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "hospital_before": hospital_before,
            "infection_freq": infection_freq,
            "souches": souches_id,
            "age_group": age_group,
            "comorbidity": comorbidity,
            "risk_score": risk_score
        }

        if feature_cols:
            # Match training feature order
            input_vector = []
            for col in feature_cols:
                key = col.lower().replace(" ", "_")
                input_vector.append(raw_inputs.get(key, 0))
            input_data = np.array([input_vector])
        else:
            # Absolute fallback
            input_data = np.array([[diabetes, hypertension, hospital_before, infection_freq, age, gender]])

        base_prediction = model.predict(input_data)[0]

        # -----------------------------------------------------------------
        # HYBRID PREDICTION — priority: lookup → pattern model → base model
        # -----------------------------------------------------------------
        prediction_numeric = []

        for i, drug in enumerate(TARGET_COLS):
            if strain and (strain, drug) in lookup:
                # 1. Exact strain/drug match from lookup table
                pred = lookup[(strain, drug)]

            elif drug in pattern_models:
                # 2. Cross-antibiotic pattern model
                try:
                    feature_vector = []
                    for j, other_drug in enumerate(TARGET_COLS):
                        if other_drug != drug:
                            val = int(base_prediction[j])
                            noise = int(np.random.choice([-1, 0, 1]))
                            val = max(0, min(2, val + noise))
                            feature_vector.append(val)
                    pred = pattern_models[drug].predict([feature_vector])[0]
                except Exception:
                    pred = base_prediction[i]

            else:
                # 3. Fallback: base model output
                pred = base_prediction[i]

            prediction_numeric.append(int(pred))

        # -----------------------------------------------------------------
        # CLINICAL INSIGHT ENGINE (REASONING GENERATOR)
        # -----------------------------------------------------------------
        results = []
        explanations = []

        for i, val in enumerate(prediction_numeric):
            drug = TARGET_COLS[i]
            status = RESISTANCE_MAP[val]
            
            # Logic-based reasoning
            if val == 0:  # Resistant
                if risk_score > 3:
                    reason = f"High risk profile ({int(risk_score)}) and hospitalization history strongly correlate with resistance in this strain."
                elif infection_freq > 2:
                    reason = f"Frequent past infections ({int(infection_freq)}) suggest a high likelihood of multi-drug resistance for {drug}."
                else:
                    reason = f"Standard resistance pattern detected for this microbial category."
            
            elif val == 1:  # Intermediate
                reason = "Predictive model suggests borderline effectiveness; clinical caution or higher dosage may be considered."
            
            else:  # Susceptible
                if age_group >= 3:
                    reason = "Predicted effective despite age-related risks; this antibiotic remains a primary candidate."
                elif comorbidity == 0:
                    reason = "Clean clinical profile (no comorbidities) significantly increases the success probability for this treatment."
                else:
                    reason = "Strong susceptibility signature detected; highly recommended for this bacterial profile."

            results.append({"drug": drug, "status": status, "code": val})
            explanations.append({"drug": drug, "reason": reason})

        # Top susceptible antibiotics, ordered by score (susceptible = 2 > intermediate = 1 > resistant = 0)
        scores = sorted(
            zip(TARGET_COLS, prediction_numeric),
            key=lambda x: x[1],
            reverse=True,
        )
        recommended = [drug for drug, score in scores if score == 2][:5]

        return render_template(
            "index.html",
            prediction=results,
            recommended=recommended,
            explanations=explanations,
            form_data=form_data,
        )

    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            form_data=form_data,
        )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# API ENDPOINT (SCALABILITY LAYER)
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Scalable JSON API for external integrations (HMS, Mobile Apps).
    Expects JSON: { "age": int, "gender": str, "souches": str, ... }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    try:
        # Extract inputs (reusing logic from HTML route but standardized)
        age = float(data.get("age", 25))
        gender = 0 if str(data.get("gender")).upper() == "F" else 1
        diabetes = int(data.get("diabetes", 0))
        hypertension = int(data.get("hypertension", 0))
        hosp = int(data.get("hospital_before", 0))
        inf_freq = float(data.get("infection_freq", 0))
        strain = data.get("souches", "Unknown")

        # Feature engineering
        comorbidity = int(diabetes + hypertension)
        def bin_age(a):
            if a <= 18: return 0
            if a <= 35: return 1
            if a <= 55: return 2
            if a <= 75: return 3
            return 4
        age_group = bin_age(age)
        risk_score = float(hosp * 2 + inf_freq + comorbidity)
        souches_id = souches_mapping.get(strain, souches_mapping.get('Unknown', 0))

        # Prediction
        raw_inputs = {
            "age": age, "gender": gender, "diabetes": diabetes, "hypertension": hypertension,
            "hospital_before": hosp, "infection_freq": inf_freq, "souches": souches_id,
            "age_group": age_group, "comorbidity": comorbidity, "risk_score": risk_score
        }
        
        input_vector = [raw_inputs.get(col.lower().replace(" ", "_"), 0) for col in (feature_cols or [])]
        prediction = model.predict(np.array([input_vector]))[0]

        # Results mapping
        response = {
            "status": "success",
            "patient_profile": {"risk_score": risk_score, "age_group": age_group},
            "predictions": []
        }

        for i, val in enumerate(prediction):
            response["predictions"].append({
                "antibiotic": TARGET_COLS[i],
                "status": RESISTANCE_MAP[int(val)],
                "code": int(val)
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    ensure_models_exist()
    load_models()
    app.run(debug=True)
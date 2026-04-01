from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None
    print("Warning: Model not found. Please run 'python src/train.py' first.")

# Target columns exactly from model training
TARGET_COLS = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
               'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']

# EXPLICIT MAPPING
# Ensures output translates the model prediction accurately to UI strings
# 2: Susceptible, 1: Intermediate, 0: Resistant
RESISTANCE_MAP = {
    0: "Resistant",
    1: "Intermediate",
    2: "Susceptible"
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, form_data={})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", prediction=None, form_data={}, error="ML Model .pkl is missing! Train the model first.")

    form_data = {
        "age": request.form.get("age", ""),
        "gender": request.form.get("gender", ""),
        "diabetes": request.form.get("diabetes", ""),
        "hypertension": request.form.get("hypertension", ""),
        "hospital_before": request.form.get("hospital_before", ""),
        "infection_freq": request.form.get("infection_freq", "")
    }

    try:
        # 1. Parse & Validate Form Inputs
        age = float(request.form.get("age", 25))
        if age < 0 or age > 100:
            return render_template("index.html", prediction=None, form_data=form_data, error="Please enter a valid age between 0 and 100.")
            
        infection_freq = float(request.form.get("infection_freq", 0))
        # The 0-20 bound reflects the realistic extremes of the clinical dataset scope. 
        # Frequencies structurally beyond 20 likely introduce immense outlier skew to probability logic.
        if infection_freq < 0 or infection_freq > 20:
            return render_template("index.html", prediction=None, form_data=form_data, error="Please enter a realistic number of past infections (0 to 20).")
        
        # Parse the rest
        gender = float(request.form.get("gender", 0))
        diabetes = float(request.form.get("diabetes", 0))
        hypertension = float(request.form.get("hypertension", 0))
        hospital_before = float(request.form.get("hospital_before", 0))
        
        # Pack form data securely
        raw_inputs = {
            "age": age,
            "gender": gender,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "hospital_before": hospital_before,
            "infection_freq": infection_freq
        }

        # 2. Package Input for Model 
        # Safely enforce mathematically identical feature ordering by loading training topology
        try:
            # Dynamically fetch the ['Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq', 'Age', 'Gender'] array
            feature_cols = joblib.load("models/feature_cols.pkl")
            # Convert training title case to form lower case for reliable dictionary matching
            ordered_input = [raw_inputs[col.lower().replace(" ", "_")] for col in feature_cols]
            input_data = np.array([ordered_input])
        except Exception:
            # Fallback if models/feature_cols.pkl wasn't built yet
            input_data = np.array([[diabetes, hypertension, hospital_before, infection_freq, age, gender]])
        
        # 3. Model Predicts 15 targets simultaneously
        prediction_numeric = model.predict(input_data)[0]
        
        # 4. Map the numerical outputs to human readable format
        results = []
        recommended = []
        
        for i, val in enumerate(prediction_numeric):
            drug_name = TARGET_COLS[i]
            status = RESISTANCE_MAP.get(int(val), "Unknown")
            
            results.append({
                "drug": drug_name, 
                "status": status, 
                "code": int(val) 
            })
            
            # 2 represents 'Susceptible' mathematically now
            if int(val) == 2:  
                recommended.append(drug_name)

        return render_template("index.html", prediction=results, recommended=recommended, form_data=form_data, error=None)

    except Exception as e:
        return render_template("index.html", prediction=None, form_data=form_data, error=f"Processing Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# Codecure AI - Antibiotic Resistance Predictor 🧬

## Problem Statement 🚨
Antibiotic resistance is one of the most significant public health threats globally. When a patient arrives with a bacterial infection, conducting physical susceptibility tests to sequence exactly which antibiotics work can take days. This delay leads to broader prescriptions, worsening multiresistance.

## Solution Overview 🚀
**Codecure** is an end-to-end, production-ready AI healthcare system utilizing a **Multi-Output Random Forest Machine Learning Model** wrapped in a sleek **Flask Backend**. 

Instead of waiting days, doctors can input 6 clinical datapoints (Age, Gender, Diabetes, Hypertension, Hospitalization history, Infection frequency) into the beautiful Glassmorphism user interface. The AI instantaneously analyzes the non-linear clinical correlations within the data and outputs a 15-level prediction array, telling the doctor explicitly which of the 15 major antibiotics are mathematically **Susceptible (Recommended)**, **Intermediate**, or **Resistant**.

## Tech Stack 💻
*   **Machine Learning**: `scikit-learn`, `pandas`, `numpy` (RandomForestClassifier + MultiOutputClassifier)
*   **Backend Server**: `Flask` (Python)
*   **Frontend UI**: Pure `HTML5` + `CSS3` (Zero JS framework overhead, dynamic glassmorphism design).

## Installation Steps ⚙️

1.  **Clone this Repository** to your local machine.
2.  **Open the terminal** and CD into the `Codecure` directory.
3.  **Install Required Packages:**
    ```bash
    pip install pandas numpy scikit-learn flask joblib
    ```

## How to Run the Project 🏃‍♂️

1.  **Train the Machine Learning Model First!**
    Since the repository does not include the huge `.pkl` file (to save space), you must generate it first using the cleaned dataset. Run:
    ```bash
    python src/train.py
    ```
    *This parses the data, trains the multi-output tree algorithms, and dynamically stores `model.pkl` in the `models/` directory.*

2.  **Boot the Flask Server:**
    ```bash
    python app.py
    ```

3.  **Access the Dashboard:**
    Open your internet browser and navigate to `http://localhost:5000` or `http://127.0.0.1:5000`

## Folder Structure 📂
```
Codecure/
├── data/
│   └── Cleaned_Bacteria_Dataset.csv   # The preprocessed, fully numeric dataset
├── models/
│   ├── model.pkl                      # Generated ML model bundle
│   └── feature_cols.pkl               # Saved structural array topology
├── src/
│   ├── clean.py                       # The script built to preprocess the raw data
│   └── train.py                       # Fits RandomForest to X vs y Targets
├── templates/
│   └── index.html                     # Frontend GUI structure
├── static/
│   └── style.css                      # Frontend design aesthetic
├── app.py                             # Flask Routing Backend
└── README.md                          # You are here!
```

The model is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Results are calculated for each antibiotic and averaged across all predictions.

Average Performance:
- Accuracy: ~72%
- Precision: ~68%
- Recall: ~72%
- F1 Score: ~70%

This ensures the model is properly validated before deployment.


## Future Improvements 🔮
1.  Connect to an SQL database to pull real-time, live hospital datapoints instead of static CSVs.
2.  Incorporate a Large Language Model (LLM) route to summarize the findings for the patient in non-medical jargon.
3.  Deploy securely to AWS EC2 or Heroku.

## Demo Overview 🎥

This system allows users to:

1. Enter patient clinical details (Age, Gender, Diabetes, etc.)
2. Predict antibiotic resistance for 15 antibiotics
3. View results as:
   - Resistant ❌
   - Intermediate ⚠️
   - Susceptible ✅
4. Get recommended antibiotics (only susceptible ones)

Example Workflow:
- Input: 45-year-old male, diabetic, prior infections
- Output: List of safe and unsafe antibiotics

## Model Details 🧠

- Model Used: RandomForestClassifier (MultiOutputClassifier)
- Input Features: 6 clinical parameters
- Output: 15 antibiotic resistance predictions
- Evaluation:
  - Exact Match Accuracy (strict)
  - Average Accuracy across targets

Why this model?
- Handles tabular data well
- Captures non-linear relationships
- Supports multi-output prediction

## Limitations ⚠️

- Model is trained on historical dataset and may not generalize perfectly
- Does not include lab-based bacterial genetic data
- Predictions are advisory and should not replace clinical judgment


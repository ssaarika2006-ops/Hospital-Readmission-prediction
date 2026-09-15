import os
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hospital_readmission_model.pkl"
model = joblib.load(MODEL_PATH)

patient = pd.DataFrame([{
    "age": 62,
    "length_of_stay": 6,
    "previous_admissions": 2,
    "number_of_medications": 7,
    "chronic_conditions": 3,
    "emergency_visit": 1,
    "followup_scheduled": 0
}])

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("Predicted readmission:", "Yes" if prediction == 1 else "No")
print(f"Model probability of readmission: {probability:.2%}")
print("Educational demo only; not for clinical decisions.")

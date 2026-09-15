import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.3);
    }
    .metric-banner {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .clinical-notice {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Hospital 30-Day Readmission Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict post-discharge hospital readmission probability using <b>RandomForestClassifier</b> trained on patient demographics, comorbidities, and clinical indicators.</div>', unsafe_allow_html=True)

st.markdown('<div class="clinical-notice">⚠️ <b>Educational & Decision-Support Demo:</b> Intended for healthcare analytics workflows. Does not replace professional clinical diagnosis.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models" / "hospital_readmission_model.pkl"
csv_path = BASE_DIR / "data" / "hospital_readmission.csv"
chart_path = BASE_DIR / "outputs" / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🩺 Patient Admission & Readmission Risk", "📈 Feature Importance & Clinical Drivers", "📋 Historical Patient Inpatient Data"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Patient Clinical Profile")
        
        preset = st.selectbox(
            "⚡ Quick Clinical Scenario Preset",
            ["Custom Clinical Case", "🚨 High-Risk Elderly Comorbid Patient", "🩺 Routine Post-Op (Follow-up Scheduled)", "🟢 Young Acute Recovered Patient"]
        )
        
        if preset == "🚨 High-Risk Elderly Comorbid Patient":
            def_age, def_stay, def_prev, def_meds, def_cond, def_emg, def_fol = 74, 9, 3, 11, 4, 1, 0
        elif preset == "🩺 Routine Post-Op (Follow-up Scheduled)":
            def_age, def_stay, def_prev, def_meds, def_cond, def_emg, def_fol = 52, 4, 1, 5, 1, 0, 1
        elif preset == "🟢 Young Acute Recovered Patient":
            def_age, def_stay, def_prev, def_meds, def_cond, def_emg, def_fol = 28, 2, 0, 2, 0, 0, 1
        else:
            def_age, def_stay, def_prev, def_meds, def_cond, def_emg, def_fol = 62, 6, 2, 7, 3, 1, 0
            
        with st.form("patient_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Patient Age (Years)", 18, 100, def_age)
                length_of_stay = st.slider("Length of Stay (Days)", 1, 30, def_stay)
                previous_admissions = st.slider("Previous Admissions (Past 12 Mos)", 0, 10, def_prev)
                chronic_conditions = st.slider("Chronic Conditions / Comorbidities", 0, 8, def_cond)
            with col_b:
                number_of_medications = st.slider("Number of Prescribed Medications", 1, 25, def_meds)
                emergency_visit = st.radio("Admitted via Emergency Department?", [1, 0], index=0 if def_emg == 1 else 1, format_func=lambda x: "Yes (Emergency)" if x == 1 else "No (Elective / Transfer)")
                followup_scheduled = st.radio("Post-Discharge Follow-up Scheduled?", [1, 0], index=0 if def_fol == 1 else 1, format_func=lambda x: "Yes (Scheduled)" if x == 1 else "No (Pending)")
                
            submit_btn = st.form_submit_button("🚀 Evaluate 30-Day Readmission Risk", use_container_width=True)
            
    with col_result:
        st.subheader("Readmission Risk Assessment")
        if submit_btn:
            patient_df = pd.DataFrame([{
                "age": age,
                "length_of_stay": length_of_stay,
                "previous_admissions": previous_admissions,
                "number_of_medications": number_of_medications,
                "chronic_conditions": chronic_conditions,
                "emergency_visit": emergency_visit,
                "followup_scheduled": followup_scheduled
            }])
            
            pred = model.predict(patient_df)[0]
            prob = model.predict_proba(patient_df)[0][1]
            
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Readmission Classification</div>
                    <div class="metric-banner">🚨 HIGH RISK OF READMISSION</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Predicted Probability: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Readmission Classification</div>
                    <div class="metric-banner">🩺 LOW RISK OF READMISSION</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Predicted Probability: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📊 Risk Stratification")
            st.progress(float(prob))
            
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Readmission Likelihood", f"{prob:.1%}")
            col_r2.metric("Safe Discharge Likelihood", f"{(1 - prob):.1%}")
            
            st.markdown("### 📋 Recommended Care Coordination Interventions")
            interventions = []
            if followup_scheduled == 0:
                interventions.append("⚠️ **Schedule Primary Care / Specialist Visit:** Ensure outpatient appointment is booked within 7 days of discharge.")
            if number_of_medications >= 6:
                interventions.append("💊 **Medication Reconciliation:** Polypharmacy detected (>5 active meds). Inpatient pharmacist consult recommended prior to release.")
            if emergency_visit == 1:
                interventions.append("📞 **Post-Discharge Outreach:** Case manager outreach phone call within 48 hours to confirm symptom stability.")
            if chronic_conditions >= 3:
                interventions.append("🫀 **Chronic Disease Management:** Enroll patient in home health monitoring / chronic condition nurse navigator program.")
                
            if interventions:
                for item in interventions:
                    st.write(item)
            else:
                st.success("✅ Routine discharge protocol standard procedures apply.")
                
            with st.expander("🔍 Patient Inpatient Clinical Payload"):
                st.json(patient_df.to_dict(orient="records")[0])
        else:
            st.info("👈 Enter patient admission indicators and click **'Evaluate 30-Day Readmission Risk'**.")

with tab2:
    st.subheader("Model Feature Importance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Feature Importance for Hospital Readmission Risk", use_container_width=True)
    else:
        st.info("Visual plot will appear after running train_model.py")

with tab3:
    st.subheader("Inpatient Training Dataset (hospital_readmission.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", f"{len(df):,}")
        col2.metric("Readmission Rate", f"{df['readmitted'].mean():.1%}")
        col3.metric("Avg Length of Stay", f"{df['length_of_stay'].mean():.1f} days")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Dataset not found.")

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load model
model = joblib.load('heart_model_v2.pkl')

st.set_page_config(page_title="Heart Health AI Pro", layout="wide") # Wide layout for better visuals
st.title("❤️ Pro Heart Disease Predictor")

# Sidebar for Disclaimer
st.sidebar.info("⚠️ **Disclaimer:** This tool is for educational purposes and based on statistical patterns. It is NOT a substitute for professional medical advice.")

# ... (Previous Input Code remains same) ...
# (Assuming variables are already defined as per previous step)

if st.button("Predict Heart Risk"):
    features = [
        high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
        phys_activity, fruits, veggies, hvy_alcohol, any_healthcare,
        no_doc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
        sex, age, edu, income
    ]
    
    feature_names = model.get_booster().feature_names
    df_input = pd.DataFrame([features], columns=feature_names)
    
    prob = model.predict_proba(df_input)[0][1]
    
    # --- VISUAL 1: Gauge Meter ---
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Heart Disease Risk Score (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

    # --- VISUAL 2: Feature Importance (Local) ---
    st.write("### 🔍 Key Factors Contributing to Your Score:")
    # Simple logic to show which user inputs were high risk
    risk_factors = []
    if high_bp == 1: risk_factors.append("High Blood Pressure")
    if smoker == 1: risk_factors.append("Smoking History")
    if bmi > 27: risk_factors.append("High BMI")
    
    if risk_factors:
        st.warning(f"Note: Your {', '.join(risk_factors)} are significant risk indicators.")
    else:
        st.success("Great! Your primary health indicators look stable.")

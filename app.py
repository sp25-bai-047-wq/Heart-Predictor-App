import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('heart_model_v2.pkl')

st.title("❤️ Pro Heart Disease Predictor")
st.write("Please provide your health details in simple English.")

col1, col2 = st.columns(2)

with col1:
    # Human-friendly labels, but we store 1 or 0
    high_bp_raw = st.selectbox("Do you have High Blood Pressure?", ["No", "Yes"])
    high_bp = 1 if high_bp_raw == "Yes" else 0

    high_chol_raw = st.selectbox("Do you have High Cholesterol?", ["No", "Yes"])
    high_chol = 1 if high_chol_raw == "Yes" else 0

    smoker_raw = st.selectbox("Have you smoked 100+ cigarettes in your life?", ["No", "Yes"])
    smoker = 1 if smoker_raw == "Yes" else 0

    stroke_raw = st.selectbox("Have you ever had a Stroke?", ["No", "Yes"])
    stroke = 1 if stroke_raw == "Yes" else 0

    # BMI remains a number as it's a value
    bmi = st.number_input("Your BMI (Body Mass Index)", value=25.0)

with col2:
    sex_raw = st.selectbox("Gender", ["Female", "Male"])
    sex = 1 if sex_raw == "Male" else 0

    # Age categories are tricky, let's make them readable
    age_map = {
        "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, 
        "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
        "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
    }
    age_label = st.selectbox("Select your Age Group", list(age_map.keys()))
    age = age_map[age_label]

    gen_hlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    gen_hlth_label = st.selectbox("How is your General Health?", list(gen_hlth_map.keys()))
    gen_hlth = gen_hlth_map[gen_hlth_label]

# ... baqi inputs bhi isi tarah 'Yes/No' mein convert kar lein

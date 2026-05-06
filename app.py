
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Model load karein
model = joblib.load('heart_model.pkl')

st.set_page_config(page_title="Heart Health Predictor", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.write("Fill in the details below to check heart health status.")

# Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    
    with col2:
        sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)

    submit = st.form_submit_button("Predict Result")

if submit:
    # Input data ko array mein convert karna (Humne 13 features use kiye the training mein)
    # Baqi features ko default values de dete hain fillhal testing ke liye
    features = [age, sex, cp, trestbps, chol, 0, 1, thalach, exang, oldpeak, 1, 0, 2]
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! Probability: {probability*100:.2f}%")
        st.write("Please consult a doctor soon.")
    else:
        st.success(f"✅ Low Risk! Probability: {probability*100:.2f}%")
        st.write("Your heart health seems normal.")

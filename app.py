import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the new XGBoost model
# Make sure the filename matches exactly what you uploaded to GitHub
model = joblib.load('heart_model_v2.pkl')

st.set_page_config(page_title="Heart Health AI Pro", layout="centered")

st.title("❤️ Pro Heart Disease Predictor")
st.write("This model is trained on **253,680 records** using XGBoost for high reliability.")

st.markdown("### Patient Health Indicators")
st.write("Please fill in the following details:")

# Creating columns for a better layout
col1, col2 = st.columns(2)

with col1:
    high_bp = st.selectbox("High Blood Pressure?", [0, 1], help="0 = No, 1 = Yes")
    high_chol = st.selectbox("High Cholesterol?", [0, 1])
    chol_check = st.selectbox("Cholesterol Check in last 5 years?", [0, 1])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    smoker = st.selectbox("Have you smoked 100+ cigarettes?", [0, 1])
    stroke = st.selectbox("Ever had a Stroke?", [0, 1])
    diabetes = st.selectbox("Diabetes Status", [0, 1, 2], help="0=No, 1=Pre, 2=Yes")

with col2:
    phys_activity = st.selectbox("Physical Activity in last 30 days?", [0, 1])
    fruits = st.selectbox("Eat Fruit daily?", [0, 1])
    veggies = st.selectbox("Eat Veggies daily?", [0, 1])
    hvy_alcohol = st.selectbox("Heavy Alcohol Consumer?", [0, 1])
    gen_hlth = st.slider("General Health Scale", 1, 5, 3, help="1=Excellent, 5=Poor")
    ment_hlth = st.slider("Days of poor mental health (last 30 days)", 0, 30, 0)
    phys_hlth = st.slider("Days of physical illness (last 30 days)", 0, 30, 0)

# More inputs to match all 21 features
diff_walk = st.selectbox("Difficulty Walking/Climbing stairs?", [0, 1])
sex = st.selectbox("Sex", [0, 1], help="0=Female, 1=Male")
age = st.slider("Age Category (1=18-24, 13=80+)", 1, 13, 5)
education = st.slider("Education Level (1-6)", 1, 6, 4)
income = st.slider("Income Level (1-8)", 1, 8, 5)
no_doc_bc_cost = st.selectbox("Couldn't see doctor due to cost?", [0, 1])
any_healthcare = st.selectbox("Do you have healthcare coverage?", [0, 1])

# Create feature list in EXACT order as trained
features = [
    high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
    phys_activity, fruits, veggies, hvy_alcohol, any_healthcare,
    no_doc_bc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
    sex, age, education, income
]

if st.button("Predict Heart Risk"):
    # Convert to DataFrame
    df_input = pd.DataFrame([features], columns=model.get_booster().feature_names)
    
    # Prediction
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High Risk!** Probability: {prob*100:.2f}%")
        st.write("Please consult a doctor for a professional check-up.")
    else:
        st.success(f"✅ **Low Risk!** Probability: {prob*100:.2f}%")
        st.write("Maintain your healthy lifestyle!")
  

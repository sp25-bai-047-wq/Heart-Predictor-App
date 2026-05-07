import streamlit as st
import joblib
import pandas as pd

# Load the XGBoost model
model = joblib.load('heart_model_v2.pkl')

st.set_page_config(page_title="Heart Health AI Pro", layout="centered")
st.title("❤️ Pro Heart Disease Predictor")
st.write("Trained on **253,680 records** using XGBoost for high accuracy.")

st.markdown("### Please enter your details:")

col1, col2 = st.columns(2)

with col1:
    high_bp = 1 if st.selectbox("High Blood Pressure?", ["No", "Yes"]) == "Yes" else 0
    high_chol = 1 if st.selectbox("High Cholesterol?", ["No", "Yes"]) == "Yes" else 0
    chol_check = 1 if st.selectbox("Cholesterol Check (Last 5 Years)?", ["No", "Yes"]) == "Yes" else 0
    bmi = st.number_input("Your BMI (Body Mass Index)", value=25.0)
    smoker = 1 if st.selectbox("Smoked 100+ cigarettes in life?", ["No", "Yes"]) == "Yes" else 0
    stroke = 1 if st.selectbox("Ever had a Stroke?", ["No", "Yes"]) == "Yes" else 0
    
    # Diabetes mapping
    diab_map = {"No": 0, "Pre-diabetes": 1, "Yes": 2}
    diabetes = diab_map[st.selectbox("Diabetes Status", list(diab_map.keys()))]
    
    phys_activity = 1 if st.selectbox("Physical Activity (Last 30 days)?", ["No", "Yes"]) == "Yes" else 0
    fruits = 1 if st.selectbox("Eat Fruits daily?", ["No", "Yes"]) == "Yes" else 0
    veggies = 1 if st.selectbox("Eat Veggies daily?", ["No", "Yes"]) == "Yes" else 0

with col2:
    hvy_alcohol = 1 if st.selectbox("Heavy Alcohol Consumer?", ["No", "Yes"]) == "Yes" else 0
    any_healthcare = 1 if st.selectbox("Have Healthcare Coverage?", ["No", "Yes"]) == "Yes" else 0
    no_doc_cost = 1 if st.selectbox("Skipped Doctor due to Cost?", ["No", "Yes"]) == "Yes" else 0
    
    # Health scales
    gen_hlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    gen_hlth = gen_hlth_map[st.selectbox("General Health Status", list(gen_hlth_map.keys()))]
    
    ment_hlth = st.slider("Poor Mental Health Days (Last 30)", 0, 30, 0)
    phys_hlth = st.slider("Poor Physical Health Days (Last 30)", 0, 30, 0)
    diff_walk = 1 if st.selectbox("Difficulty Walking/Climbing?", ["No", "Yes"]) == "Yes" else 0
    
    sex = 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
    
    # Age category mapping
    age_map = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13}
    age = age_map[st.selectbox("Age Group", list(age_map.keys()))]
    
    edu = st.slider("Education Level (1-6)", 1, 6, 4)
    income = st.slider("Income Level (1-8)", 1, 8, 5)

# THE IMPORTANT SUBMISSION BUTTON
if st.button("Predict Heart Risk"):
    # Features must match the exact 21 columns of your XGBoost model
    features = [
        high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
        phys_activity, fruits, veggies, hvy_alcohol, any_healthcare,
        no_doc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
        sex, age, edu, income
    ]
    
    # Convert to DataFrame
    df_input = pd.DataFrame([features], columns=model.get_booster().feature_names)
    
    # Prediction
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High Risk Predicted!** (Probability: {prob*100:.1f}%)")
        st.write("Please consult a cardiologist for further evaluation.")
    else:
        st.success(f"✅ **Low Risk Predicted!** (Probability: {prob*100:.1f}%)")
        st.write("Keep maintaining your healthy lifestyle!")

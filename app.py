import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# --- 1. Load Model ---
try:
    model = joblib.load('heart_model_v2.pkl')
except:
    st.error("Model file not found. Please ensure 'heart_model_v2.pkl' is in GitHub.")

# --- 2. Page Config ---
st.set_page_config(page_title="Heart Health AI Pro", layout="wide")
st.title("❤️ Pro Heart Disease Predictor")

# Sidebar Disclaimer
st.sidebar.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. It is NOT a substitute for professional medical advice.")

st.markdown("### Please fill in your health details:")

# --- 3. Inputs ---
col1, col2 = st.columns(2)

with col1:
    high_bp = 1 if st.selectbox("High Blood Pressure?", ["No", "Yes"]) == "Yes" else 0
    high_chol = 1 if st.selectbox("High Cholesterol?", ["No", "Yes"]) == "Yes" else 0
    chol_check = 1 if st.selectbox("Cholesterol Check (Last 5 Years)?", ["No", "Yes"]) == "Yes" else 0
    bmi = st.number_input("Your BMI", value=25.0)
    smoker = 1 if st.selectbox("Smoked 100+ cigarettes?", ["No", "Yes"]) == "Yes" else 0
    stroke = 1 if st.selectbox("Ever had a Stroke?", ["No", "Yes"]) == "Yes" else 0
    diabetes = st.selectbox("Diabetes (0=No, 1=Pre, 2=Yes)", [0, 1, 2])
    phys_activity = 1 if st.selectbox("Physical Activity?", ["No", "Yes"]) == "Yes" else 0
    fruits = 1 if st.selectbox("Eat Fruits daily?", ["No", "Yes"]) == "Yes" else 0
    veggies = 1 if st.selectbox("Eat Veggies daily?", ["No", "Yes"]) == "Yes" else 0

with col2:
    hvy_alcohol = 1 if st.selectbox("Heavy Alcohol?", ["No", "Yes"]) == "Yes" else 0
    any_healthcare = 1 if st.selectbox("Healthcare Coverage?", ["No", "Yes"]) == "Yes" else 0
    no_doc_cost = 1 if st.selectbox("Skipped Doctor due to Cost?", ["No", "Yes"]) == "Yes" else 0
    gen_hlth = st.slider("General Health (1=Ex, 5=Poor)", 1, 5, 3)
    ment_hlth = st.slider("Mental Health Days (Last 30)", 0, 30, 0)
    phys_hlth = st.slider("Physical Health Days (Last 30)", 0, 30, 0)
    diff_walk = 1 if st.selectbox("Difficulty Walking?", ["No", "Yes"]) == "Yes" else 0
    sex = 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
    age = st.slider("Age Category (1-13)", 1, 13, 5)
    edu = st.slider("Education (1-6)", 1, 6, 4)
    income = st.slider("Income (1-8)", 1, 8, 5)

# --- 4. Prediction Logic ---
if st.button("Predict Heart Risk"):
    features = [
        high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
        phys_activity, fruits, veggies, hvy_alcohol, any_healthcare,
        no_doc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
        sex, age, edu, income
    ]
    
    # Matching feature names
    df_input = pd.DataFrame([features], columns=model.get_booster().feature_names)
    
    # Get Probability
    prob = model.predict_proba(df_input)[0][1]
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Percentage", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig)
    # --- VISUAL 2: Local Risk Factors Text ---
    st.write("---") # Horizontal line for separation
    st.write("### 🔍 Key Factors Contributing to Your Score:")
    
    # We identify which 'Yes' inputs are known high-risk factors
    major_risks = []
    if high_bp == 1: major_risks.append("High Blood Pressure")
    if smoker == 1: major_risks.append("Smoking History")
    if stroke == 1: major_risks.append("Previous Stroke")
    if diabetes == 2: major_risks.append("Diabetes")
    if bmi > 28: major_risks.append(f"High BMI ({bmi})")
    
    if major_risks:
        st.warning(f"Note: Your {', '.join(major_risks)} are significant risk indicators based on our data.")
        # Minimalist Recommendations
        st.write("#### 💡 Next Steps:")
        if high_bp == 1 or high_chol == 1:
            st.write("- Focus on a heart-healthy diet (low sodium).")
        if smoker == 1:
            st.write("- Consult resources for quitting smoking.")
        if phys_activity == 0:
            st.write("- Aim for at least 150 minutes of moderate activity per week.")
    else:
        st.success("Great! Based on your inputs, your primary health indicators look stable.")
    
    if prob > 0.5:
        st.error(f"⚠️ **High Risk!** Score: {prob*100:.1f}%")
        st.balloons() # Thora twist: High risk par balloons nahi, warning honi chahiye but test it!
    else:
        st.success(f"✅ **Low Risk!** Score: {prob*100:.1f}%")
        st.balloons()
    # --- STEP: PDF REPORT GENERATION ---
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Heart Health Assessment Report', 0, 1, 'C')
            self.ln(5)

    def generate_pdf(prob_val, risk_txt, user_data):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt=f"Risk Score: {prob_val*100:.1f}%", ln=1)
        pdf.cell(200, 10, txt=f"Assessment: {risk_txt}", ln=1)
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Input Details:", ln=1)
        pdf.set_font("Arial", size=10)
        for key, value in user_data.items():
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=1)
            
        return pdf.output(dest='S').encode('latin-1')

    # Data for PDF
    user_summary = {
        "BMI": bmi,
        "High BP": "Yes" if high_bp == 1 else "No",
        "Smoker": "Yes" if smoker == 1 else "No",
        "Age Category": age,
        "General Health": gen_hlth
    }
    
    risk_label = "High Risk" if prob > 0.5 else "Low Risk"
    pdf_bytes = generate_pdf(prob, risk_label, user_summary)

    st.download_button(
        label="📥 Download Full Health Report (PDF)",
        data=pdf_bytes,
        file_name="Heart_Health_Report.pdf",
        mime="application/pdf"
    )

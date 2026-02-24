import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="Loan Approval System", layout="wide")

# 1. Load the saved model and scaler
# Using relative paths; ensure you run streamlit from the project root
model = joblib.load('models/loan_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 2. Sidebar for System Information
with st.sidebar:
    st.title("System Overview")
    st.write("Module: Introduction to Artificial Intelligence")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Accuracy: 98%")
    st.write("---")
    st.write("Status: Operational")

# 3. Main Interface Header
st.title("Loan Approval Prediction System")
st.write("Manually enter the applicant details below for assessment.")
st.write("---")

# 4. Input Fields (Manual Text Input)
st.subheader("Section 1: Applicant Profile")
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.text_input("Number of Dependents", value="0")
    education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employment Status", options=["Yes", "No"])

with col2:
    income_annum = st.text_input("Annual Income", value="0")
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)

st.write("---")
st.subheader("Section 2: Loan and Assets")
a_col1, a_col2 = st.columns(2)

with a_col1:
    loan_amount = st.text_input("Requested Loan Amount", value="0")
    loan_term = st.text_input("Loan Term (Years)", value="0")
    bank_assets = st.text_input("Total Bank Asset Value", value="0")

with a_col2:
    residential_assets = st.text_input("Residential Assets Value", value="0")
    commercial_assets = st.text_input("Commercial Assets Value", value="0")
    luxury_assets = st.text_input("Luxury Assets Value", value="0")

# 5. Prediction Logic
if st.button("Generate Eligibility Report"):
    try:
        # Convert text inputs to correct types
        dep_val = int(no_of_dependents)
        edu_val = 1 if education == "Graduate" else 0
        emp_val = 1 if self_employed == "Yes" else 0
        inc_val = float(income_annum)
        amt_val = float(loan_amount)
        term_val = int(loan_term)
        res_val = float(residential_assets)
        com_val = float(commercial_assets)
        lux_val = float(luxury_assets)
        bnk_val = float(bank_assets)
        
        # ARRANGE DATA IN THE EXACT ORDER OF YOUR DATAFRAME COLUMNS:
        # no_of_dependents, education, self_employed, income_annum, loan_amount, 
        # loan_term, cibil_score, residential_assets_value, commercial_assets_value, 
        # luxury_assets_value, bank_asset_value
        input_data = np.array([[
            dep_val, edu_val, emp_val, inc_val, amt_val, 
            term_val, cibil_score, res_val, com_val, lux_val, bnk_val
        ]])
        
        # Apply the scaler (StandardScaler)
        input_scaled = scaler.transform(input_data)
        
        # Prediction
        prediction = model.predict(input_scaled)
        
        # Results Display
        st.write("---")
        st.subheader("Assessment Result")
        if prediction[0] == 1:
            st.success("Result: LOAN APPROVED")
            st.write("The applicant meets the required financial thresholds and credit criteria.")
        else:
            st.error("Result: LOAN REJECTED")
            st.write("The system has identified high-risk factors (e.g., low CIBIL score or high debt-to-income ratio).")
            
    except ValueError:
        st.error("Error: Please ensure all manual inputs are valid numbers.")
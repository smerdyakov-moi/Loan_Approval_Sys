import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler
model = joblib.load('models/loan_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 2. Set up the Web Page UI
st.title("Loan Approval Prediction System")
st.write("Enter the applicant's details below to check loan eligibility.")

# 3. Create Input Fields for the User (Based on your CSV columns)
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0)
    loan_term = st.number_input("Loan Term (Years)", min_value=0, max_value=30)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets = st.number_input("Luxury Assets Value", min_value=0)
    bank_assets = st.number_input("Bank Asset Value", min_value=0)

# 4. Process Inputs when the button is clicked
if st.button("Predict Loan Status"):
    # Convert text inputs to numbers (just like we did in preprocessing)
    edu_val = 1 if education == "Graduate" else 0
    emp_val = 1 if self_employed == "Yes" else 0
    
    # Arrange data in the exact order the model expects
    input_data = np.array([[no_of_dependents, edu_val, emp_val, income_annum, 
                            loan_amount, loan_term, cibil_score, residential_assets, 
                            commercial_assets, luxury_assets, bank_assets]])
    
    # Apply the same scaling we used during training
    input_scaled = scaler.transform(input_data)
    
    # Make Prediction
    prediction = model.predict(input_scaled)
    
    # Show Result
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")
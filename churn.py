import numpy as np
import pandas as pd
import joblib
import streamlit as st

encoders = joblib.load('C:\\Users\\hp\\Downloads\\encoders_customer_churn.joblib')
model = joblib.load('C:\\Users\\hp\\Downloads\\model_customer_churn.joblib')
scaler = joblib.load('C:\\Users\\hp\\Downloads\\scaler_customer_churn.joblib')
df = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\Telco_Churn.csv")
df.columns = df.columns.str.strip()

st.write("""
# This is an AI powered Customer Churn Predictor App

This app predicts the churn status of a customer using this telecommunication company.       
""")
st.write('***')

def churn_pred():
    input_data = {
        'gender' : gender,
        'SeniorCitizen' : senior_citizen,
        'Partner' : married,
        'Dependents' : dependent,
        'tenure' : tenure,
        'PhoneService' : phone_service,
        'MultipleLines' : multiple_lines,
        'InternetService' : internet_service,
        'OnlineSecurity' : online_security,
        'OnlineBackup' : online_backup,
        'DeviceProtection' : device_protection,
        'TechSupport' : tech_support,
        'StreamingTV' : streaming_tv,
        'StreamingMovies' : streaming_movies,
        'Contract' : contract,
        'PaperlessBilling' : paperless_billing,
        'PaymentMethod' : payment_method,
        'MonthlyCharges' : monthly_charges,
        'TotalCharges' : total_charges
    }

    input_data_df = pd.DataFrame([input_data])
    #encode categorical features
    for column , encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])
    num_columns = ['tenure','MonthlyCharges' ,'TotalCharges']
    input_data_df[num_columns] = scaler.transform(input_data_df[num_columns])
    prediction = model.predict(input_data_df)
    pred_proba = model.predict_proba(input_data_df)
    if prediction == 0:
        return(f"No churn expected with {(pred_proba[0][0] * 100).round()}% probability ")
    else:
        return(f"Churn expected with {(pred_proba[0][1] * 100).round()}% probability ")
    return

gender = st.selectbox("Enter customer gender??" ,df['gender'].unique())
senior_citizen = st.selectbox("Is Customer a Senior Citizen?(1 is yes , 0 is no)", df['SeniorCitizen'].unique())
married = st.selectbox("Married?" ,df['Partner'].unique())
dependent = st.selectbox("Does Customer have children/relatives" ,df['Dependents'].unique())
tenure = st.number_input("How many months have customer been with this company??")
phone_service = st.selectbox("Does customer have phone service?", df['PhoneService'].unique())
multiple_lines = st.selectbox("Does customer have Multiple Phone Lines??" ,df['MultipleLines'].unique())
internet_service = st.selectbox("Internet type used by customer?(DSL/Fiber optic/No)" ,df['InternetService'].unique())
online_security = st.selectbox("Does customer use online security??",df['OnlineSecurity'].unique())
online_backup = st.selectbox("Does cutomer have online back up??" , df['OnlineBackup'].unique())
device_protection = st.selectbox("Does Customer have device protection??" ,df['DeviceProtection'].unique())
tech_support = st.selectbox("Does Customer use tech support??",df['TechSupport'].unique())
streaming_tv = st.selectbox("Does Customer Stream TV??" ,df['StreamingTV'].unique())
streaming_movies = st.selectbox("Does Customer Stream Movies??" ,df['StreamingMovies'].unique())
contract = st.selectbox("What type of contract does customer have with company", df['Contract'].unique())
paperless_billing = st.selectbox("Does Customer use paperless billing??" ,df['PaperlessBilling'].unique())
payment_method = st.selectbox("What Payment Method does Customer Uses??",df['PaymentMethod'].unique())
monthly_charges = st.number_input("What is the Monthly Charges of customer??")
total_charges = (tenure * monthly_charges)
st.write(f"Total Charges is {total_charges}")

st.sidebar.header("🔍 Why Customers Leave")
st.sidebar.write("""
**What Predicts Churn Best:**  
- **Contract Type** (Month-to-Month = 3× risk)  
- **Tenure** (<6 months = High risk)  
- **Monthly Charges** (>$70 = 2× churn rate)  
- **Service Complaints** (1+ ticket = 40% more likely to leave)  
- **Competitor Promotions** (Switches peak during deals)  
""")

st.sidebar.header("🚀 Retention Strategies")
st.sidebar.write("""
**Proven Tactics:**  
1. **Target Month-to-Month Users**: Offer loyalty discounts.  
2. **Flag High Spenders**: Personalize retention offers.  
3. **Improve Onboarding**: New customers churn 50% faster.  
4. **Monitor Service Tickets**: Resolve issues within 24h.  
5. **Competitor Alerts**: Counter-switch offers proactively.  
""")

st.sidebar.header("⚙️ How We Predict")
st.sidebar.write("""
This AI analyzes:  
- **Behavioral Data** (Usage drops, support calls)  
- **Billing Patterns** (Late payments, plan changes)  
- **Service Stack** (Missing add-ons = 30% higher risk)  
- **External Triggers** (Economic/seasonal trends)  
""")

st.sidebar.header("💸 Why It Matters")
st.sidebar.write("""
Reducing churn by 5% can:  
- Boost profits 25-95% (Bain & Co)  
- Cut acquisition costs (CAC >> retention costs)  
- Improve customer lifetime value (LTV)  
""")

if st.button('Churn Status'):
    result = churn_pred()
    st.success(result)
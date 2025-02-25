import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the trained model (Booster object)
model = joblib.load('ML_Model FINAL.pkl')  # Ensure correct path

# Load and preprocess the HELOC dataset for the dashboard
@st.cache_resource
def load_heloc_data():
    df = pd.read_csv("heloc_dataset_v1.csv")  # Ensure correct path

    # Remove rows with missing values (-9)
    df = df[~df.isin([-9]).any(axis=1)]

    # Replace -7 with group means based on RiskPerformance
    group_means = df.replace(-7, np.nan).groupby('RiskPerformance').mean()

    def impute_with_group_mean(row):
        for col in df.columns:
            if col == 'RiskPerformance' or not np.issubdtype(df[col].dtype, np.number):
                continue
            if row[col] == -7:
                row[col] = group_means.loc[row['RiskPerformance'], col]
        return row

    df = df.apply(impute_with_group_mean, axis=1)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Select only necessary columns
    selected_features = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate', 
                         'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days']

    return df[selected_features], df  # Return both the selected features and full dataset

heloc_data, full_data = load_heloc_data()

# Streamlit UI with Tabs
tab1, tab2 = st.tabs(["📊 HELOC Predictor", "📈 Dashboard"])

# -----------------------  TAB 1: HELOC PREDICTOR -----------------------
with tab1:
    st.title("🏦 HELOC Eligibility Predictor")
    st.write("📊 Enter your financial details to check HELOC eligibility.")

    # API Key Input (hidden for privacy)
    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

    # Define input fields (only 5 selected features)
    feature_order = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate', 
                     'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days']

    user_input = {
        'ExternalRiskEstimate': st.slider("📈 External Risk Estimate (Credit Score)", 0, 100, 50),
        'MSinceMostRecentDelq': st.slider("📅 Months Since Most Recent Delinquency", 0, 100, 10),
        'MaxDelqEver': st.slider("⚠️ Maximum Delinquency Ever Recorded (0 = None, 1-8 = Increasing Severity)", 0, 8, 2),
        'PercentTradesNeverDelq': st.slider("📉 Percentage of Trades Never Delinquent", 0, 100, 80),
        'MSinceMostRecentInqexcl7days': st.slider("🔍 Months Since Most Recent Inquiry (Excluding Last 7 Days)", 0, 100, 10)
    }

    # Convert user input into DataFrame
    input_data = pd.DataFrame([user_input])[feature_order]

    # Predict eligibility
    if st.button("📌 Check Eligibility"):
        try:
            # Convert input data to DMatrix
            input_dmatrix = xgb.DMatrix(input_data)

            # Get probability score from the model
            probability = model.predict(input_dmatrix)[0]

            # Convert probability into a binary classification
            threshold = 0.5  # Adjust if needed
            prediction = 1 if probability >= threshold else 0

            # Show results
            if prediction == 1:
                st.success(f"✅ Eligible for HELOC! Approval Probability: {probability:.2%}")
                explanation_prompt = "This applicant is eligible for a HELOC. Can you provide financial advice and responsible loan usage tips?"
            else:
                st.error(f"❌ Not Eligible. Approval Probability: {probability:.2%}")
                explanation_prompt = f"""
                This applicant was denied a HELOC loan.
                - External Risk Estimate: {user_input['ExternalRiskEstimate']}
                - Most Recent Delinquency: {user_input['MSinceMostRecentDelq']} months ago
                - Maximum Delinquency Severity: {user_input['MaxDelqEver']}
                - Percentage of Non-Delinquent Trades: {user_input['PercentTradesNeverDelq']}%
                - Months Since Last Credit Inquiry (Excl. Last 7 Days): {user_input['MSinceMostRecentInqexcl7days']} months

                Based on these factors, provide possible reasons for rejection and actionable suggestions to improve eligibility.
                """

            # Get GPT Explanation if API key is provided
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": explanation_prompt}]
                    )
                    st.write("💡 **AI Financial Insights:**")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"⚠️ OpenAI API Error: {str(e)}")

        except Exception as e:
            st.error(f"⚠️ Model Prediction Error: {str(e)}")

# -----------------------  TAB 2: DASHBOARD -----------------------
with tab2:
    st.title("📈 HELOC Data Dashboard")
    st.write("🔍 Explore HELOC applicant data and analyze trends.")

    # Sidebar Filters
    st.sidebar.header("📊 Filter Data")
    min_credit = st.sidebar.slider("Min External Risk Estimate", int(full_data['ExternalRiskEstimate'].min()), int(full_data['ExternalRiskEstimate'].max()), 50)
    delinquency_filter = st.sidebar.slider("Max Months Since Delinquency", int(full_data['MSinceMostRecentDelq'].min()), int(full_data['MSinceMostRecentDelq'].max()), 50)

    # Filter dataset based on user selections
    filtered_data = full_data[
        (full_data['ExternalRiskEstimate'] >= min_credit) &
        (full_data['MSinceMostRecentDelq'] <= delinquency_filter)
    ]

    # Show Dataset Preview
    st.write("📋 Filtered HELOC Data")
    st.dataframe(filtered_data)

    # AI Chat Feature for Additional Analysis
    if api_key:
        st.write("💬 **Ask AI for Insights on the Data**")
        user_question = st.text_area("❓ Ask anything about the HELOC dataset:")

        if st.button("🤖 Get AI Response"):
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": user_question}]
                )
                st.write("💬 **AI Response:**")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"⚠️ OpenAI API Error: {str(e)}")

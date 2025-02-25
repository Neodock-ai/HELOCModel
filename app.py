import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
                         'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days', 'RiskPerformance']

    return df[selected_features], df  # Return both selected features and full dataset

heloc_data, full_data = load_heloc_data()

# **Initialize global variables**
probability = None  
user_input = None  

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
            else:
                st.error(f"❌ Not Eligible. Approval Probability: {probability:.2%}")

        except Exception as e:
            st.error(f"⚠️ Model Prediction Error: {str(e)}")

# -----------------------  TAB 2: DASHBOARD (DYNAMIC INSIGHTS) -----------------------
with tab2:
    st.title("📈 Personalized HELOC Insights Dashboard")
    st.write("🔍 Explore insights based on your entered data.")

    if probability is not None:
        # **DYNAMIC KPIs** based on user input
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Approval Probability", f"{probability:.2%}")
        with col2:
            st.metric("💳 Your Credit Score", f"{user_input['ExternalRiskEstimate']}")
        with col3:
            st.metric("⚠️ Delinquency Severity", f"{user_input['MaxDelqEver']}")

        # **Visualizing User Input Compared to Historical Data**
        st.subheader("📊 Your Input vs. HELOC Trends")

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(full_data["ExternalRiskEstimate"], bins=20, kde=True, label="Dataset Distribution", ax=ax)
        ax.axvline(user_input['ExternalRiskEstimate'], color='red', linestyle='--', label="Your Score")
        ax.legend()
        st.pyplot(fig)

        # **Approval Rate Comparison**
        st.subheader("📊 Loan Approval Rate Based on Your Input")
        approval_rate = full_data[(full_data["ExternalRiskEstimate"] >= user_input['ExternalRiskEstimate']) & 
                                  (full_data["MSinceMostRecentDelq"] <= user_input['MSinceMostRecentDelq'])]["RiskPerformance"].value_counts(normalize=True) * 100

        fig, ax = plt.subplots()
        sns.barplot(x=approval_rate.index, y=approval_rate.values, palette="coolwarm", ax=ax)
        ax.set_ylabel("Approval Rate (%)")
        ax.set_xlabel("Risk Performance")
        ax.set_title("Loan Approval Rate Based on Input")
        st.pyplot(fig)

    else:
        st.warning("⚠️ No prediction made yet. Enter details in the HELOC Predictor tab and check eligibility.")

    # AI Chat Feature for Dashboard Insights
    if api_key:
        st.write("💬 **Ask AI for Insights on Your Data**")
        user_question = st.text_area("❓ Ask anything about your HELOC eligibility:")

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

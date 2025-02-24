import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('ML_Model FINAL.pkl')  # Ensure correct path
data = pd.read_csv('heloc_dataset_v1.csv')  # Load HELOC dataset for dashboard

# Create Tabs for Predictor & Dashboard
tab1, tab2 = st.tabs(["📊 HELOC Predictor", "📈 Dashboard"])

# -----------------------  TAB 1: HELOC PREDICTOR -----------------------
with tab1:
    st.title("🏦 HELOC Eligibility Predictor")
    st.write("📊 Enter your financial details to check HELOC eligibility.")

    # API Key Input (hidden for privacy)
    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

    # Define input fields (only 5 selected features)
    feature_order = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate', 'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days']
    
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

            # AI Explanation
            if api_key:
                explanation_prompt = f"""
                The applicant has the following details:
                - External Risk Estimate: {user_input['ExternalRiskEstimate']}
                - Most Recent Delinquency: {user_input['MSinceMostRecentDelq']} months ago
                - Maximum Delinquency Severity: {user_input['MaxDelqEver']}
                - Percentage of Non-Delinquent Trades: {user_input['PercentTradesNeverDelq']}%
                - Months Since Last Credit Inquiry (Excl. Last 7 Days): {user_input['MSinceMostRecentInqexcl7days']} months
                
                Provide financial insights based on these factors.
                """
                try:
                    client = openai.OpenAI(api_key=api_key)
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
    min_credit = st.sidebar.slider("Min External Risk Estimate", int(data['ExternalRiskEstimate'].min()), int(data['ExternalRiskEstimate'].max()), 50)
    delinquency_filter = st.sidebar.slider("Max Months Since Delinquency", int(data['MSinceMostRecentDelq'].min()), int(data['MSinceMostRecentDelq'].max()), 50)

    # Filter dataset based on user selections
    filtered_data = data[
        (data['ExternalRiskEstimate'] >= min_credit) &
        (data['MSinceMostRecentDelq'] <= delinquency_filter)
    ]

    # Layout: Two Columns for Visualizations
    col1, col2 = st.columns(2)

    # First Visualization: Approval Rate Distribution
    with col1:
        st.subheader("📊 Approval Rate Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_data['ExternalRiskEstimate'], bins=20, kde=True, ax=ax, color="blue")
        ax.set_xlabel("Credit Score (External Risk Estimate)")
        ax.set_ylabel("Frequency")
        ax.set_title("HELOC Approvals vs. Credit Score")
        st.pyplot(fig)

    # Second Visualization: Delinquency Impact
    with col2:
        st.subheader("📉 Delinquency vs. Approval Rate")
        fig, ax = plt.subplots()
        sns.boxplot(x=filtered_data['MSinceMostRecentDelq'], y=filtered_data['PercentTradesNeverDelq'], ax=ax, palette="coolwarm")
        ax.set_xlabel("Months Since Most Recent Delinquency")
        ax.set_ylabel("Percentage of Non-Delinquent Trades")
        ax.set_title("Impact of Delinquency on Approval")
        st.pyplot(fig)

    # AI-Generated Insights Section
    if api_key:
        st.write("🧠 **AI-Generated Insights on HELOC Data**")
        analysis_prompt = f"""
        The dataset contains information on HELOC applicants.
        - The distribution of external risk estimate (credit scores) vs approval rate is shown.
        - The impact of delinquency history on approval rates is visualized.
        
        Generate insights on how these factors influence HELOC approvals.
        """
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ OpenAI API Error: {str(e)}")

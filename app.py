import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb

# Load the trained model (Booster object)
model = joblib.load('/content/ML_Model FINAL.pkl')  # Ensure correct path

# Streamlit UI
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

# Convert user input into DataFrame and ensure feature order consistency
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
            st.success(f"✅ Congratulations! You are eligible for a HELOC. Approval Probability: {probability:.2%}")
            explanation_prompt = "This applicant is eligible for a HELOC. Can you provide financial advice and responsible loan usage tips?"
        else:
            st.error(f"❌ Unfortunately, you are not eligible for a HELOC. Approval Probability: {probability:.2%}")
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
                st.write("💡 **AI Explanation:**")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"⚠️ OpenAI API Error: {str(e)}")
    except Exception as e:
        st.error(f"⚠️ Model Prediction Error: {str(e)}")

# 🔹 Chat with GPT for More Explanation
if api_key:
    st.write("💬 **Ask AI for Further Explanation**")
    user_question = st.text_area("❓ Ask about your eligibility results:", "")

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

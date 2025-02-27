import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Custom Styling for Dark Mode & Background
st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background: url('https://source.unsplash.com/1600x900/?finance,technology') no-repeat center fixed;
        background-size: cover;
    }

    /* Title Styling */
    h1 {
        color: #ffffff;
        text-align: center;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: rgba(30, 30, 30, 0.8) !important;
    }

    /* Custom Buttons */
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #ffffff;
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: 2px solid #ffffff;
    }

    /* Custom Metrics */
    div[data-testid="metric-container"] {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 8px;
        color: white !important;
        text-align: center;
    }

    /* Input Sliders */
    .stSlider>div>div>div {
        background-color: #007bff !important;
    }

    </style>
""", unsafe_allow_html=True)

# Load the trained model (Booster object)
model = joblib.load('ML_Model FINAL.pkl')

# Load and preprocess the HELOC dataset
@st.cache_resource
def load_heloc_data():
    df = pd.read_csv("heloc_dataset_v1.csv")

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
    df = df.drop_duplicates()

    selected_features = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate',
                         'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days', 'RiskPerformance']

    return df[selected_features], df  # Return selected features & full dataset

heloc_data, full_data = load_heloc_data()

# **Initialize session state if not set**
if "probability" not in st.session_state:
    st.session_state.probability = None
    st.session_state.prediction = None
    st.session_state.user_input = None

# Streamlit UI with Tabs
tab1, tab2 = st.tabs(["📊 HELOC Predictor", "📈 Dashboard"])

# -----------------------  TAB 1: HELOC PREDICTOR -----------------------
with tab1:
    st.title("🏦 HELOC Eligibility Predictor")
    st.write("📊 Enter your financial details to check HELOC eligibility.")

    # API Key Input (hidden for privacy)
    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

    feature_order = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate',
                     'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days']

    user_input = {
        'ExternalRiskEstimate': st.slider("📈 External Risk Estimate (Credit Score)", 0, 100, 50),
        'MSinceMostRecentDelq': st.slider("📅 Months Since Most Recent Delinquency", 0, 100, 10),
        'MaxDelqEver': st.slider("⚠️ Maximum Delinquency Ever Recorded (0 = None, 1-8 = Increasing Severity)", 0, 8, 2),
        'PercentTradesNeverDelq': st.slider("📉 Percentage of Trades Never Delinquent", 0, 100, 80),
        'MSinceMostRecentInqexcl7days': st.slider("🔍 Months Since Most Recent Inquiry (Excluding Last 7 Days)", 0, 100, 10)
    }

    input_data = pd.DataFrame([user_input])[feature_order]

    if st.button("📌 Check Eligibility"):
        try:
            input_dmatrix = xgb.DMatrix(input_data)
            probability = model.predict(input_dmatrix)[0]
            threshold = 0.5  
            prediction = "Eligible" if probability >= threshold else "Not Eligible"

            st.session_state.probability = probability
            st.session_state.prediction = prediction
            st.session_state.user_input = user_input

            if prediction == "Eligible":
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
                """

            # Provide feedback regardless of OpenAI API access
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
                    st.info("💡 **Default Feedback:**")
                    st.write(explanation_prompt)
            else:
                st.info("💡 **Default Feedback:**")
                st.write(explanation_prompt)

        except Exception as e:
            st.error(f"⚠️ Model Prediction Error: {str(e)}")

# -----------------------  TAB 2: DASHBOARD (DYNAMIC INSIGHTS) -----------------------
with tab2:
    st.title("📈 Personalized HELOC Insights Dashboard")
    st.write("🔍 Explore insights based on your entered data.")

    if st.session_state.probability is not None:
        user_input = st.session_state.user_input

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Approval Probability", f"{st.session_state.probability:.2%}")
        with col2:
            st.metric("💳 Your Credit Score", f"{user_input['ExternalRiskEstimate']}")
        with col3:
            st.metric("⚠️ Delinquency Severity", f"{user_input['MaxDelqEver']}")

        st.subheader("📊 Your Credit Score vs Dataset Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(full_data["ExternalRiskEstimate"], bins=20, kde=True, label="Dataset Distribution", ax=ax)
        ax.axvline(user_input['ExternalRiskEstimate'], color='red', linestyle='--', label="Your Score")
        ax.legend()
        st.pyplot(fig)

        # ------------------ Additional Graphs and Analysis ------------------
        st.subheader("Correlation Heatmap of Financial Metrics")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        corr = full_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Credit Score vs. Non-Delinquent Trades")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="ExternalRiskEstimate", y="PercentTradesNeverDelq", hue="RiskPerformance", data=full_data, ax=ax3)
        ax3.axvline(user_input['ExternalRiskEstimate'], color='red', linestyle='--', label="Your Credit Score")
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("Boxplot: Credit Score Distribution by Risk Performance")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="RiskPerformance", y="ExternalRiskEstimate", data=full_data, ax=ax4)
        st.pyplot(fig4)

        st.subheader("Distribution: Months Since Most Recent Delinquency")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.histplot(full_data["MSinceMostRecentDelq"], bins=20, kde=True, ax=ax5)
        ax5.axvline(user_input['MSinceMostRecentDelq'], color='red', linestyle='--', label="Your Value")
        ax5.legend()
        st.pyplot(fig5)
    else:
        st.warning("⚠️ No prediction made yet. Enter details in the HELOC Predictor tab and check eligibility.")

    if api_key:
        st.write("💬 **Ask AI for Insights on Your Data**")
        user_question = st.text_area("❓ Ask anything about your HELOC eligibility:")

        if st.button("🤖 Get AI Response"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"User inputs: {st.session_state.user_input}, {user_question}"}]
            )
            st.write("💬 **AI Response:**")
            st.write(response.choices[0].message.content)

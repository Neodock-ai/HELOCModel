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
    api_key = st.text_input("🔑 Enter your OpenAI API Key (Optional)", type="password")

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

            # Display clear colored outcome messages
            if prediction == "Eligible":
                st.success(f"✅ Eligible for HELOC! Approval Probability: {probability:.2%}")
            else:
                st.error(f"❌ Not Eligible. Approval Probability: {probability:.2%}")

            # Detailed default feedback messages with actionable insights
            if prediction == "Eligible":
                default_feedback = f"""
**Outcome:** Eligible for HELOC  
Your approval probability is **{probability:.2%}**.

**Observations:**  
- **Credit Health:** Your External Risk Estimate is **{user_input['ExternalRiskEstimate']}**, which is a strong indicator.  
- **Delinquency:** You have had no recent delinquencies (last **{user_input['MSinceMostRecentDelq']}** months) with a maximum severity of **{user_input['MaxDelqEver']}**.  
- **Trade Performance:** **{user_input['PercentTradesNeverDelq']}%** of your trades have never been delinquent.  
- **Inquiries:** Your recent credit inquiry history (past **{user_input['MSinceMostRecentInqexcl7days']}** months) is healthy.

**Advice:** Continue maintaining your good credit habits. Regular monitoring and prompt payment will help you keep your eligibility intact.
"""
            else:
                default_feedback = f"""
**Outcome:** Not Eligible for HELOC  
Your approval probability is **{probability:.2%}**.

**Observations:**  
- **Credit Score:** Your External Risk Estimate is **{user_input['ExternalRiskEstimate']}**, which may be below the desired threshold.  
- **Delinquency:** It has been **{user_input['MSinceMostRecentDelq']}** months since your last delinquency, with a severity level of **{user_input['MaxDelqEver']}**.  
- **Trade Performance:** **{user_input['PercentTradesNeverDelq']}%** of your trades are free from delinquency, which is a positive sign.  
- **Inquiries:** Your record shows **{user_input['MSinceMostRecentInqexcl7days']}** months since your last inquiry.

**Advice:** Consider improving your credit by addressing past delinquencies and reducing new credit inquiries. Consulting with a financial advisor might help create a tailored improvement plan.
"""

            # Fetch AI-enhanced feedback if API key is provided
            if api_key:
                try:
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": default_feedback}]
                    )
                    ai_feedback = response.choices[0].message.content
                    st.markdown("### 💡 AI Financial Insights")
                    st.write(ai_feedback)
                except Exception as e:
                    st.error(f"⚠️ OpenAI API Error: {str(e)}")
                    st.markdown("### 📋 Default Financial Feedback")
                    st.write(default_feedback)
            else:
                st.markdown("### 📋 Default Financial Feedback")
                st.write(default_feedback)

        except Exception as e:
            st.error(f"⚠️ Model Prediction Error: {str(e)}")

# -----------------------  TAB 2: DASHBOARD (DYNAMIC INSIGHTS) -----------------------
with tab2:
    st.title("📈 Personalized HELOC Insights Dashboard")
    st.write("🔍 Explore a comprehensive analysis of your financial inputs compared to historical data.")

    if st.session_state.probability is not None:
        user_input = st.session_state.user_input

        # KPI Section: Key Performance Indicators
        st.subheader("Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Approval Probability", f"{st.session_state.probability:.2%}")
        with col2:
            st.metric("Credit Score", f"{user_input['ExternalRiskEstimate']}")
        with col3:
            st.metric("Delinquency (Months)", f"{user_input['MSinceMostRecentDelq']}")
        with col4:
            st.metric("Trade Health (%)", f"{user_input['PercentTradesNeverDelq']}%")

        st.markdown("---")
        
        # Comparative Analysis: Display dataset statistics for key metrics
        numeric_cols = ['ExternalRiskEstimate', 'MSinceMostRecentDelq', 'MaxDelqEver', 'PercentTradesNeverDelq', 'MSinceMostRecentInqexcl7days']
        stats = full_data[numeric_cols].agg(['mean', 'median']).T.reset_index().rename(columns={'index': 'Metric'})
        
        st.subheader("Your Metrics vs. Dataset Averages")
        st.dataframe(stats.style.format({"mean": "{:.2f}", "median": "{:.2f}"}))
        
        # Bar Chart: Compare user values against dataset means
        st.subheader("Comparison Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        x_labels = numeric_cols
        user_values = [user_input[col] for col in numeric_cols]
        dataset_means = [full_data[col].mean() for col in numeric_cols]
        x = np.arange(len(x_labels))
        width = 0.35

        ax.bar(x - width/2, user_values, width, label='Your Value', color='teal')
        ax.bar(x + width/2, dataset_means, width, label='Dataset Mean', color='gray')
        ax.set_ylabel('Value')
        ax.set_title('Your Metrics vs. Dataset Mean')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()
        st.pyplot(fig)

        # Additional Visualizations for common understanding

        # Histogram: Credit Score Distribution
        st.subheader("Credit Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(full_data["ExternalRiskEstimate"], bins=20, kde=True, ax=ax2, color='skyblue')
        ax2.axvline(user_input['ExternalRiskEstimate'], color='red', linestyle='--', label="Your Score")
        ax2.set_xlabel("External Risk Estimate")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)

        # Scatter Plot: Delinquency vs. Credit Score
        st.subheader("Delinquency vs. Credit Score")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x="ExternalRiskEstimate", y="MSinceMostRecentDelq", hue="RiskPerformance", data=full_data, ax=ax3, palette='viridis')
        ax3.axvline(user_input['ExternalRiskEstimate'], color='red', linestyle='--', label="Your Credit Score")
        ax3.axhline(user_input['MSinceMostRecentDelq'], color='orange', linestyle='--', label="Your Delinquency")
        ax3.set_xlabel("External Risk Estimate")
        ax3.set_ylabel("Months Since Most Recent Delinquency")
        ax3.legend()
        st.pyplot(fig3)

        # Boxplot: Credit Score Distribution by Risk Performance
        st.subheader("Credit Score Distribution by Risk Performance")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="RiskPerformance", y="ExternalRiskEstimate", data=full_data, ax=ax4, palette='pastel')
        ax4.set_xlabel("Risk Performance")
        ax4.set_ylabel("External Risk Estimate")
        st.pyplot(fig4)
    else:
        st.warning("⚠️ No prediction made yet. Enter your details in the HELOC Predictor tab and check eligibility.")

    if api_key:
        st.markdown("### 💬 Ask AI for Additional Insights")
        user_question = st.text_area("❓ Ask anything about your HELOC eligibility:")
        if st.button("🤖 Get AI Response"):
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": f"User inputs: {st.session_state.user_input}, {user_question}"}]
                )
                st.markdown("#### 💬 AI Response:")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"⚠️ OpenAI API Error: {str(e)}")

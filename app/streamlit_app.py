import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Term Deposit Predictor | 10 ML Models",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Glassmorphism, Modern Typography & Gradients)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .main-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(165, 180, 252, 0.4);
        color: #c7d2fe;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 18px;
    }

    .metric-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
    }

    .metric-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .result-card-yes {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        animation: fadeIn 0.4s ease-in-out;
    }

    .result-card-no {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        animation: fadeIn 0.4s ease-in-out;
    }

    .badge-yes {
        background: #10b981;
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }

    .badge-no {
        background: #ef4444;
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Loader (Cached for instant response)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models and metadata...")
def load_artifacts():
    # Model paths
    model_path = os.path.join("models", "classification_models.joblib")
    meta_path = os.path.join("models", "classification_metadata.json")
    data_path = os.path.join("data", "bank-full.csv")

    # Fallback paths if run from different cwd
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "models", "classification_models.joblib")
        meta_path = os.path.join("..", "models", "classification_metadata.json")
        data_path = os.path.join("..", "data", "bank-full.csv")

    models = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    df = pd.read_csv(data_path, sep=";")
    return models, metadata, df

try:
    models, metadata, df = load_artifacts()
    benchmark_df = pd.DataFrame.from_dict(metadata["benchmark_metrics"], orient="index")
    benchmark_df.index.name = "Algorithm"
    benchmark_df.reset_index(inplace=True)
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure `models/classification_models.joblib` exists.")
    st.stop()

# -----------------------------------------------------------------------------
# Header Hero Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-hero">
    <div class="hero-badge">23CSE301 Capstone Project • Classification Track</div>
    <h1 style="margin: 0 0 8px 0; font-size: 32px; color: white;">🏦 Bank Marketing Term Deposit Predictor</h1>
    <p style="margin: 0; font-size: 15px; color: #cbd5e1; max-width: 820px; line-height: 1.5;">
        Predict whether a bank client will subscribe to a long-term deposit (<code style="color: #a5b4fc; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">y: yes / no</code>) based on client demographics, financial indicators, and telemarketing contact dynamics across <b>10 trained classification algorithms</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Model Selection & Presets
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Algorithm Selection")

model_options = list(models.keys())
selected_model_name = st.sidebar.selectbox(
    "Choose Classifier",
    options=model_options,
    index=model_options.index("Gradient Boosting") if "Gradient Boosting" in model_options else 0,
    help="Select which of the 10 trained pipelines to use for making predictions."
)

# Display model test metrics from benchmark
if selected_model_name in metadata["benchmark_metrics"]:
    m_info = metadata["benchmark_metrics"][selected_model_name]
    t_info = metadata["tuning_summary"].get(selected_model_name, {})
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 12px 14px; margin: 12px 0;">
        <div style="font-size: 11px; text-transform: uppercase; color: #6366f1; font-weight: 700; letter-spacing: 0.05em;">Held-Out Test Performance</div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 13px;">
            <span>Accuracy: <b>{m_info['Accuracy']*100:.1f}%</b></span>
            <span>ROC-AUC: <b>{m_info['ROC-AUC']:.3f}</b></span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 13px;">
            <span>Precision: <b>{m_info['Precision']:.3f}</b></span>
            <span>Weighted F1: <b>{m_info['F1-Score (Weighted)']:.3f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.caption(f"**Hyperparameters:** `{t_info.get('Best Parameters', 'Default')}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Quick Preset Profiles")
preset = st.sidebar.radio(
    "Load Example Client:",
    options=["Custom Input", "🌟 High Propensity Client", "⚖️ Typical Moderate Client", "🚫 Cold Contact (Low Propensity)"],
    index=0
)

# Set defaults based on preset
if preset == "🌟 High Propensity Client":
    p_age, p_job, p_marital, p_education = 58, "retired", "married", "tertiary"
    p_default, p_balance, p_housing, p_loan = "no", 4500, "no", "no"
    p_contact, p_month, p_day, p_duration, p_campaign = "cellular", "oct", 15, 650, 1
    p_pdays, p_previous, p_poutcome = 90, 3, "success"
elif preset == "⚖️ Typical Moderate Client":
    p_age, p_job, p_marital, p_education = 39, "management", "married", "secondary"
    p_default, p_balance, p_housing, p_loan = "no", 1200, "yes", "no"
    p_contact, p_month, p_day, p_duration, p_campaign = "cellular", "may", 18, 220, 2
    p_pdays, p_previous, p_poutcome = -1, 0, "unknown"
elif preset == "🚫 Cold Contact (Low Propensity)":
    p_age, p_job, p_marital, p_education = 28, "blue-collar", "single", "secondary"
    p_default, p_balance, p_housing, p_loan = "no", 80, "yes", "yes"
    p_contact, p_month, p_day, p_duration, p_campaign = "unknown", "may", 27, 45, 6
    p_pdays, p_previous, p_poutcome = -1, 0, "unknown"
else:
    p_age, p_job, p_marital, p_education = 39, "management", "married", "tertiary"
    p_default, p_balance, p_housing, p_loan = "no", 1000, "yes", "no"
    p_contact, p_month, p_day, p_duration, p_campaign = "cellular", "may", 15, 250, 2
    p_pdays, p_previous, p_poutcome = -1, 0, "unknown"

# -----------------------------------------------------------------------------
# Main Section: Input Feature Tabs & Cards
# -----------------------------------------------------------------------------
st.markdown("### 📋 Client & Campaign Predictor Attributes")

col1, col2 = st.columns(2)

with col1:
    with st.expander("👤 1. Client Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Client Age", min_value=18, max_value=95, value=p_age, step=1)
            job_opts = sorted(df["job"].unique().tolist())
            job = st.selectbox("Job / Occupation", options=job_opts, index=job_opts.index(p_job) if p_job in job_opts else 0)
        with c2:
            marital_opts = sorted(df["marital"].unique().tolist())
            marital = st.selectbox("Marital Status", options=marital_opts, index=marital_opts.index(p_marital) if p_marital in marital_opts else 0)
            edu_opts = sorted(df["education"].unique().tolist())
            education = st.selectbox("Education Level", options=edu_opts, index=edu_opts.index(p_education) if p_education in edu_opts else 0)

    with col1:
        with st.expander("💰 2. Financial & Credit Indicators", expanded=True):
            f1, f2 = st.columns(2)
            with f1:
                balance = st.number_input("Average Yearly Balance (€)", min_value=-5000, max_value=100000, value=p_balance, step=100)
                default = st.selectbox("Credit in Default?", options=["no", "yes"], index=0 if p_default == "no" else 1)
            with f2:
                housing = st.selectbox("Has Housing Loan?", options=["no", "yes"], index=0 if p_housing == "no" else 1)
                loan = st.selectbox("Has Personal Loan?", options=["no", "yes"], index=0 if p_loan == "no" else 1)

with col2:
    with st.expander("📞 3. Current Campaign Direct Contact", expanded=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            contact_opts = sorted(df["contact"].unique().tolist())
            contact = st.selectbox("Contact Communication Type", options=contact_opts, index=contact_opts.index(p_contact) if p_contact in contact_opts else 0)
            month_opts = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            month = st.selectbox("Last Contact Month", options=month_opts, index=month_opts.index(p_month) if p_month in month_opts else 4)
            day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=p_day, step=1)
        with cc2:
            duration = st.slider("Last Call Duration (seconds)", min_value=0, max_value=3000, value=p_duration, step=10, help="Crucial predictor: Highly correlates with customer engagement.")
            campaign = st.number_input("Campaign Contacts (Calls during current campaign)", min_value=1, max_value=50, value=p_campaign, step=1)

    with col2:
        with st.expander("⏱️ 4. Previous Campaign History", expanded=True):
            p1, p2 = st.columns(2)
            with p1:
                pdays = st.number_input("Pdays (Days passed after previous campaign; -1 = not contacted)", min_value=-1, max_value=900, value=p_pdays, step=1)
                previous = st.number_input("Previous Contacts (Calls before this campaign)", min_value=0, max_value=50, value=p_previous, step=1)
            with p2:
                poutcome_opts = sorted(df["poutcome"].unique().tolist())
                poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=poutcome_opts, index=poutcome_opts.index(p_poutcome) if p_poutcome in poutcome_opts else 0)

# -----------------------------------------------------------------------------
# Decision Threshold Slider & Predict Action
# -----------------------------------------------------------------------------
st.markdown("---")
action_col1, action_col2 = st.columns([1, 2])

with action_col1:
    threshold = st.slider(
        "Decision Cutoff Threshold ($P(\\text{Yes})$)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Default is 0.50. Lowering this threshold (e.g. to 0.25) prioritizes high subscriber recall to capture more revenue at the cost of additional calls."
    )

with action_col2:
    st.write("")
    st.write("")
    predict_clicked = st.button("🚀 Predict Term Deposit Subscription Propensity", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# Prediction Execution & Results Card
# -----------------------------------------------------------------------------
if predict_clicked:
    # Build single-row DataFrame matching training feature columns
    input_data = pd.DataFrame([{
        "age": int(age),
        "job": str(job),
        "marital": str(marital),
        "education": str(education),
        "default": str(default),
        "balance": int(balance),
        "housing": str(housing),
        "loan": str(loan),
        "contact": str(contact),
        "day": int(day),
        "month": str(month),
        "duration": int(duration),
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": str(poutcome)
    }])

    selected_pipeline = models[selected_model_name]

    # Predict probability or decision score
    if hasattr(selected_pipeline, "predict_proba"):
        probs = selected_pipeline.predict_proba(input_data)[0]
        prob_no, prob_yes = float(probs[0]), float(probs[1])
        final_class = 1 if prob_yes >= threshold else 0
    elif hasattr(selected_pipeline, "decision_function"):
        score = float(selected_pipeline.decision_function(input_data)[0])
        # Approximate sigmoid conversion for display
        prob_yes = 1 / (1 + np.exp(-score))
        prob_no = 1 - prob_yes
        final_class = 1 if prob_yes >= threshold else 0
    else:
        final_class = int(selected_pipeline.predict(input_data)[0])
        prob_yes = 1.0 if final_class == 1 else 0.0
        prob_no = 1.0 - prob_yes

    st.markdown("### 🎯 Prediction Results")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if final_class == 1:
            st.markdown(f"""
            <div class="result-card-yes">
                <span class="badge-yes">✅ WILL SUBSCRIBE (YES)</span>
                <h2 style="color: #065f46; margin: 14px 0 6px 0; font-size: 26px;">High Propensity to Subscribe</h2>
                <p style="color: #047857; margin: 0; font-size: 15px;">
                    Client exhibits strong demographic and behavioral indicators favoring term deposit uptake. Model <b>{selected_model_name}</b> estimates a subscription probability of <b>{prob_yes*100:.1f}%</b> (threshold: {threshold:.2f}).
                </p>
                <hr style="border: 0; border-top: 1px solid rgba(16, 185, 129, 0.3); margin: 16px 0;">
                <div style="font-size: 13px; color: #065f46;">
                    <b>Recommended Action:</b> Prioritize for direct follow-up phone contact by senior relationship managers.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-no">
                <span class="badge-no">❌ WILL NOT SUBSCRIBE (NO)</span>
                <h2 style="color: #991b1b; margin: 14px 0 6px 0; font-size: 26px;">Low Propensity to Subscribe</h2>
                <p style="color: #b91c1c; margin: 0; font-size: 15px;">
                    Client attributes indicate unlikely term deposit uptake. Model <b>{selected_model_name}</b> estimates a subscription probability of only <b>{prob_yes*100:.1f}%</b> against the threshold of {threshold:.2f}.
                </p>
                <hr style="border: 0; border-top: 1px solid rgba(239, 68, 68, 0.3); margin: 16px 0;">
                <div style="font-size: 13px; color: #991b1b;">
                    <b>Recommended Action:</b> Deprioritize telemarketing call resources to avoid excessive outreach costs and customer attrition.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; font-size: 14px; color: #334155; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">
                Posterior Class Probability Distribution
            </div>
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #1e293b;">Deposit Subscription (Class 1)</span>
                    <span style="font-weight: 700; color: #10b981;">{prob_yes*100:.1f}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 9999px; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #10b981, #059669); width: {prob_yes*100}%; height: 100%;"></div>
                </div>
            </div>
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #1e293b;">No Subscription (Class 0)</span>
                    <span style="font-weight: 700; color: #64748b;">{prob_no*100:.1f}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 9999px; height: 10px; overflow: hidden;">
                    <div style="background: #94a3b8; width: {prob_no*100}%; height: 100%;"></div>
                </div>
            </div>
            <div style="font-size: 12px; color: #64748b; background: #f8fafc; padding: 8px 12px; border-radius: 8px; border: 1px dashed #cbd5e1;">
                ℹ️ Input passed directly through the fitted <b>{selected_model_name}</b> pipeline including <code>StandardScaler</code> and <code>OneHotEncoder</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Secondary Section: Comprehensive 10-Model Benchmark Comparison
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 10-Model Benchmark Performance Comparison")

tab1, tab2 = st.tabs(["📈 Metric Comparison Table & Charts", "💡 Key Project Insights"])

with tab1:
    col_t1, col_t2 = st.columns([1.1, 1])
    with col_t1:
        st.markdown("##### Consolidated Test Results ($N = 905$ Held-Out Instances)")
        formatted_bench = benchmark_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score (Weighted)", "ROC-AUC"]:
            formatted_bench[col] = formatted_bench[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            formatted_bench,
            column_config={
                "Algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
                "Accuracy": st.column_config.TextColumn("Accuracy"),
                "F1-Score (Weighted)": st.column_config.TextColumn("F1 (Weighted)"),
                "ROC-AUC": st.column_config.TextColumn("ROC-AUC")
            },
            hide_index=True,
            use_container_width=True
        )

    with col_t2:
        st.markdown("##### ROC-AUC by Model Family")
        chart_df = benchmark_df.sort_values(by="ROC-AUC", ascending=True)
        st.bar_chart(
            chart_df.set_index("Algorithm")["ROC-AUC"],
            color="#6366f1",
            use_container_width=True
        )

with tab2:
    st.markdown("""
    - **Top Performers**: **Bagging Classifier** (ROC-AUC: `0.907`, F1: `0.891`) and **Gradient Boosting** (ROC-AUC: `0.905`, F1: `0.885`) achieved the best overall balance between discriminatory power and weighted F1-score.
    - **Primary Discriminators**: Feature importances across all tree ensembles (Random Forest, Gradient Boosting, AdaBoost) converge on `duration`, `poutcome_success`, `balance`, `age`, and contact timing (`month`, `day`).
    - **Operational Threshold Calibration**: Because the dataset is severely imbalanced (88.3% 'no' vs 11.7% 'yes'), standard 0.50 cutoffs produce conservative predictions with higher false negatives. Adjusting the threshold to **0.25 - 0.30** maximizes subscription capture in practical telemarketing campaigns.
    """)

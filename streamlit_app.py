"""
streamlit_app.py — Self-contained Streamlit app for Streamlit Cloud deployment.
Loads the model and encoders directly (no Flask backend needed).
"""

import os
import pickle
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "salary_data.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#1f2937; margin-bottom:0.2rem; }
.sub-header  { font-size:1rem; color:#6b7280; margin-bottom:1.5rem; }
.salary-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #2E86C1 100%);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}
.salary-value { font-size:3rem; font-weight:800; letter-spacing:-1px; }
.salary-label { font-size:1rem; opacity:0.85; margin-top:0.3rem; }
div[data-testid="stSidebar"] { background:#f8fafc; }
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    return model, encoders, meta


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">💼 Salary Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Estimate annual salary based on education, experience, location, and role.</p>', unsafe_allow_html=True)

try:
    model, encoders, meta = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}. Make sure `models/` folder is committed to the repo.")
    st.stop()

classes = meta["classes"]
metrics = meta.get("metrics", {})
df = load_data()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Employee Details")
    st.markdown("---")
    education  = st.selectbox("🎓 Education Level",  classes["Education"])
    job_title  = st.selectbox("💼 Job Title",         classes["Job_Title"])
    location   = st.selectbox("📍 Location",          classes["Location"])
    gender     = st.selectbox("🧑 Gender",            classes["Gender"])
    experience = st.slider("📅 Years of Experience", 0, 40, 5)
    age        = st.slider("🎂 Age", 18, 70, 30)
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Salary", use_container_width=True, type="primary")

# ── Main columns ──────────────────────────────────────────────────────────────
col_result, col_info = st.columns([3, 2], gap="large")

with col_result:
    if predict_btn:
        # Build feature vector
        row = []
        for col in meta["feature_cols"]:
            if col in meta["categorical_cols"]:
                val = {"Education": education, "Location": location,
                       "Job_Title": job_title, "Gender": gender}[col]
                row.append(int(encoders[col].transform([val])[0]))
            else:
                row.append(float({"Experience": experience, "Age": age}[col]))

        salary = float(model.predict(np.array([row]))[0])
        importances = dict(zip(
            meta["feature_cols"],
            [round(float(v), 4) for v in model.feature_importances_]
        ))

        # Salary card
        st.markdown(f"""
        <div class="salary-card">
            <div class="salary-label">Predicted Annual Salary</div>
            <div class="salary-value">${salary:,.0f}</div>
            <div class="salary-label">Based on your profile</div>
        </div>""", unsafe_allow_html=True)

        # Feature importance chart
        imp_df = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values("Importance", ascending=True)

        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importances", color="Importance",
                     color_continuous_scale="Blues", template="plotly_white")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0),
                          coloraxis_showscale=False, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Fill in the employee details in the sidebar, then click **Predict Salary**.")

with col_info:
    st.subheader("📊 Model Performance")
    if metrics:
        m1, m2 = st.columns(2)
        m1.metric("R² Score", f"{metrics.get('r2', '—')}")
        m2.metric("Mean Abs. Error", f"${metrics.get('mae', 0):,.0f}")

    st.markdown("---")
    st.subheader("📋 Dataset Overview")
    st.markdown(f"**{len(df)} records** · **{df.shape[1]} features**")

    edu_avg = df.groupby("Education")["Salary"].mean().reset_index()
    fig_edu = px.bar(edu_avg, x="Education", y="Salary",
                     title="Avg Salary by Education",
                     color="Salary", color_continuous_scale="Viridis",
                     template="plotly_white")
    fig_edu.update_layout(height=240, margin=dict(l=0,r=0,t=40,b=0),
                          coloraxis_showscale=False, title_font_size=14)
    st.plotly_chart(fig_edu, use_container_width=True)

    job_avg = df.groupby("Job_Title")["Salary"].mean().reset_index()
    fig_job = px.pie(job_avg, names="Job_Title", values="Salary",
                     title="Salary Share by Job Title",
                     template="plotly_white", hole=0.4)
    fig_job.update_layout(height=240, margin=dict(l=0,r=0,t=40,b=0), title_font_size=14)
    st.plotly_chart(fig_job, use_container_width=True)

# ── Data explorer ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Explore the Raw Dataset"):
    fc1, fc2, fc3 = st.columns(3)
    edu_f = fc1.multiselect("Education", df["Education"].unique(), default=list(df["Education"].unique()))
    loc_f = fc2.multiselect("Location",  df["Location"].unique(),  default=list(df["Location"].unique()))
    job_f = fc3.multiselect("Job Title", df["Job_Title"].unique(), default=list(df["Job_Title"].unique()))

    filtered = df[df["Education"].isin(edu_f) & df["Location"].isin(loc_f) & df["Job_Title"].isin(job_f)]
    st.dataframe(filtered.style.format({"Salary": "${:,.2f}"}), use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} rows")

    fig_s = px.scatter(filtered, x="Experience", y="Salary", color="Education",
                       opacity=0.75, title="Experience vs Salary",
                       template="plotly_white", hover_data=["Job_Title","Location","Age"])
    fig_s.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0), title_font_size=14)
    st.plotly_chart(fig_s, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr><center style='color:#9ca3af;font-size:12px;'>Salary Prediction App · Random Forest · scikit-learn · Streamlit</center>", unsafe_allow_html=True)

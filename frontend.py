"""
frontend.py — Streamlit frontend for Salary Prediction
Run with: streamlit run frontend.py

Requires the Flask backend to be running at http://localhost:5000
(start with: python app.py)
"""

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:5000/api"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .salary-card {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .salary-value {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .salary-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    .metric-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stSidebar"] {
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_meta():
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def check_backend():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def predict(payload: dict):
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
    return r.json(), r.status_code


# ── App layout ────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">💼 Salary Prediction App</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Estimate annual salary based on education, experience, location, and role.</p>',
    unsafe_allow_html=True,
)

# Backend health banner
if not check_backend():
    st.error(
        "⚠️ Cannot reach the backend API at `http://localhost:5000`.  "
        "Start it with: **`python app.py`**"
    )
    st.stop()

meta = fetch_meta()
if "error" in meta:
    st.error(f"API error: {meta['error']}")
    st.stop()

classes = meta["classes"]
metrics = meta.get("metrics", {})

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Employee Details")
    st.markdown("---")

    education = st.selectbox("🎓 Education Level", classes["Education"])
    job_title = st.selectbox("💼 Job Title", classes["Job_Title"])
    location = st.selectbox("📍 Location", classes["Location"])
    gender = st.selectbox("🧑 Gender", classes["Gender"])
    experience = st.slider("📅 Years of Experience", min_value=0, max_value=40, value=5, step=1)
    age = st.slider("🎂 Age", min_value=18, max_value=70, value=30, step=1)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Salary", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
col_result, col_info = st.columns([3, 2], gap="large")

with col_result:
    if predict_btn:
        payload = {
            "Education": education,
            "Experience": experience,
            "Location": location,
            "Job_Title": job_title,
            "Age": age,
            "Gender": gender,
        }

        with st.spinner("Predicting…"):
            result, status = predict(payload)

        if status != 200:
            st.error(f"Prediction error: {result.get('error', 'Unknown error')}")
        else:
            salary = result["predicted_salary"]
            importances = result["feature_importances"]

            # Salary card
            st.markdown(
                f"""
                <div class="salary-card">
                    <div class="salary-label">Predicted Annual Salary</div>
                    <div class="salary-value">${salary:,.0f}</div>
                    <div class="salary-label">Based on your profile</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Feature importance chart
            imp_df = (
                pd.DataFrame(
                    {"Feature": list(importances.keys()), "Importance": list(importances.values())}
                )
                .sort_values("Importance", ascending=True)
            )

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importances",
                color="Importance",
                color_continuous_scale="Blues",
                template="plotly_white",
            )
            fig_imp.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_showscale=False,
                title_font_size=14,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

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
    try:
        df = pd.read_csv("data/salary_data.csv")

        st.markdown(f"**{len(df)} records** · **{df.shape[1]} features**")

        # Salary by Education
        edu_avg = df.groupby("Education")["Salary"].mean().reset_index()
        edu_avg.columns = ["Education", "Avg Salary"]
        fig_edu = px.bar(
            edu_avg,
            x="Education",
            y="Avg Salary",
            title="Avg Salary by Education",
            color="Avg Salary",
            color_continuous_scale="Viridis",
            template="plotly_white",
        )
        fig_edu.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
            title_font_size=14,
        )
        st.plotly_chart(fig_edu, use_container_width=True)

        # Salary by Job Title
        job_avg = df.groupby("Job_Title")["Salary"].mean().reset_index()
        job_avg.columns = ["Job Title", "Avg Salary"]
        fig_job = px.pie(
            job_avg,
            names="Job Title",
            values="Avg Salary",
            title="Salary Share by Job Title",
            template="plotly_white",
            hole=0.4,
        )
        fig_job.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            title_font_size=14,
        )
        st.plotly_chart(fig_job, use_container_width=True)

    except FileNotFoundError:
        st.warning("CSV not found at `data/salary_data.csv`.")

# ── Bottom — data explorer ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Explore the Raw Dataset"):
    try:
        df = pd.read_csv("data/salary_data.csv")

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        edu_filter = fc1.multiselect("Education", options=df["Education"].unique(), default=list(df["Education"].unique()))
        loc_filter = fc2.multiselect("Location", options=df["Location"].unique(), default=list(df["Location"].unique()))
        job_filter = fc3.multiselect("Job Title", options=df["Job_Title"].unique(), default=list(df["Job_Title"].unique()))

        filtered = df[
            df["Education"].isin(edu_filter)
            & df["Location"].isin(loc_filter)
            & df["Job_Title"].isin(job_filter)
        ]

        st.dataframe(filtered.style.format({"Salary": "${:,.2f}"}), use_container_width=True)
        st.caption(f"Showing {len(filtered)} of {len(df)} rows")

        # Scatter: Experience vs Salary
        fig_scatter = px.scatter(
            filtered,
            x="Experience",
            y="Salary",
            color="Education",
            size_max=8,
            opacity=0.75,
            title="Experience vs Salary",
            template="plotly_white",
            hover_data=["Job_Title", "Location", "Age"],
        )
        fig_scatter.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), title_font_size=14)
        st.plotly_chart(fig_scatter, use_container_width=True)

    except FileNotFoundError:
        st.warning("CSV not found at `data/salary_data.csv`.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><hr><center style='color:#9ca3af;font-size:12px;'>Salary Prediction App · Built with Streamlit, Flask & scikit-learn</center>",
    unsafe_allow_html=True,
)

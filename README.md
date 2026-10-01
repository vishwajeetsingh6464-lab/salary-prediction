# 💼 Salary Prediction App

A machine learning web application that predicts annual salary based on education, experience, location, job title, age, and gender.

**Live Demo:** *(deploy to get your link)*

---

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

1. Go to **https://github.com/new** and create a new **public** repository (e.g. `salary-prediction`)
2. Download and install Git from **https://git-scm.com/downloads**
3. Open PowerShell in the `salary_prediction` folder and run:

```bash
git init
git add .
git commit -m "Initial commit - Salary Prediction App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/salary-prediction.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/salary-prediction`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"**

✅ Your public URL will be: `https://YOUR_USERNAME-salary-prediction-streamlit-app-XXXX.streamlit.app`

---

## 🖥️ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (one-time)
python train_model.py

# Run the standalone app
streamlit run streamlit_app.py

# OR run with Flask backend (advanced)
python app.py          # Terminal 1
streamlit run frontend.py  # Terminal 2
```

---

## 📁 Project Structure

```
salary_prediction/
├── streamlit_app.py       # ★ Main app (Streamlit Cloud entry point)
├── train_model.py         # Train & save ML model
├── app.py                 # Flask REST API (local use)
├── frontend.py            # Streamlit UI with Flask backend (local use)
├── requirements.txt       # Python dependencies
├── data/
│   └── salary_data.csv    # 249 employee records
├── models/
│   ├── model.pkl          # Trained Random Forest model
│   ├── encoders.pkl       # Label encoders for categorical features
│   └── meta.json          # Feature metadata + model metrics
└── .streamlit/
    └── config.toml        # Streamlit theme & server config
```

---

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 200 |
| Train/Test Split | 80% / 20% |
| R² Score | **0.8746** |
| Mean Absolute Error | **$9,128** |
| CV R² (5-fold) | 0.817 ± 0.019 |

## 🎯 Features

| Feature | Values |
|---|---|
| Education | High School, Bachelor, Master, PhD |
| Experience | 0–40 years |
| Location | Urban, Suburban, Rural |
| Job Title | Analyst, Director, Engineer, Manager |
| Age | 18–70 |
| Gender | Male, Female |

---

## 🛠️ Tech Stack

- **ML:** scikit-learn (Random Forest), pandas, numpy
- **Frontend:** Streamlit, Plotly
- **Backend (local):** Flask, flask-cors
- **Deployment:** Streamlit Community Cloud

# ⚡ AutoML POC — Self-Serve Analytics & Modeling Engine

A Streamlit-based web application that automates:

- Data Profiling
- Data Cleaning (JSON expansion, % / currency parsing)
- KPI Detection
- Anomaly Detection
- Regression & Classification Modeling
- Cross-Validation
- Drift Detection (PSI)
- Feature Importance
- AI-powered Data Q&A (Claude API)

---

## 🚀 Live Demo

https://automl-dwtreyou58t9cpg7f4yehz.streamlit.app/

---

## 🧠 Problem Statement

Data analysts spend significant time cleaning and preparing data before modeling.

This project reduces manual effort by allowing users to:

1. Upload CSV or Excel files  
2. Automatically clean & profile data  
3. Train multiple ML models  
4. Compare performance  
5. Detect drift  
6. Ask AI questions about their dataset  

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Anthropic Claude API

---

## 📊 Features

### 🔎 Data Profiling
- Null detection
- Skew detection
- Correlation heatmaps
- Distribution plots

### 🧹 Data Cleaning
- Automatic numeric detection
- % and currency parsing
- JSON column expansion
- Duplicate removal
- Bulk column drop

### 🤖 Modeling
- Regression & Classification auto-detection
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Cross-validation
- Feature importance

### 📉 Monitoring
- PSI drift detection
- Anomaly detection (Isolation Forest, Z-score, IQR)

### 🧠 AI Q&A
- Claude-powered dataset insights
- Model explanation support

### 🏗 System Flow

User Upload  
→ Data Profiling  
→ Data Cleaning Engine  
→ Feature Engineering  
→ Model Training Layer  
→ Evaluation Engine  
→ Drift Monitoring (PSI)  
→ AI Insight Layer (Claude)  
→ Downloads (Predictions + Model Bundle)

---

## ⚙️ Installation (Local)

```bash
git clone <https://github.com/srijansrivastava11/automl>
cd automl-poc
pip install -r requirements.txt
streamlit run app.py

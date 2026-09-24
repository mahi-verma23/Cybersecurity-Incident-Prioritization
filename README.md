# 🔐 Cybersecurity Incident Prioritization
 
**Stack:** Python · Streamlit · Scikit-learn · Plotly · Pandas

---

## 📌 Project Description
Business Problem:
How can an organization prioritize cybersecurity incidents based on severity, attack patterns and potential impact so that investigation resources are allocated efficiently?  

This project builds an end-to-end cybersecurity analytics and machine learning dashboard that helps organisations **prioritise cybersecurity incidents** by severity, attack pattern, and data-exfiltration risk — ensuring investigation resources are allocated efficiently.

The dashboard covers:

- Data loading, validation, and cleaning
- Exploratory Data Analysis (EDA) with descriptive statistics
- 7+ interactive visualisations (incidents over time, attack frequency, severity heatmaps, etc.)
- 8 data-supported insights, 5 testable hypotheses, and 6 business recommendations
- A Random Forest ML model that predicts **Attack Severity** from pre-triage features (no leakage)
- Full classification evaluation (accuracy, F1, ROC-AUC, confusion matrix, feature importance)
- A clean, multi-tab Streamlit dashboard with sidebar filters

---

## 📂 Dataset

| Property | Value |
|---|---|
| File | `archive/cybersecurity_dataset.csv` |
| Rows | 20,000 cybersecurity incident records |
| Columns | Event ID, Timestamp, Source IP, Destination IP, User Agent, Attack Type, Attack Severity, Data Exfiltrated, Threat Intelligence, Response Action |
| Date Range | 2020-01-01 → 2024-01-30 |
| Source | Kaggle — [Cybersecurity Incidents Dataset](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks) |

Place the CSV at `archive/cybersecurity_dataset.csv` relative to the project root.

---

## 🛠️ Technologies

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data loading, cleaning, and feature engineering |
| `numpy` | Numerical operations |
| `plotly` | Interactive visualisations |
| `matplotlib` / `seaborn` | Static chart support |
| `scikit-learn` | ML model (Random Forest), evaluation metrics |

---

## 🚀 Setup & Run Instructions

### 1. Clone / download the project

```bash
git clone <repo-url>
cd cybersecurity-incident-prioritization
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Ensure the CSV is located at:

```
data/cybersecurity_dataset.csv
```

### 5. Run the dashboard

```bash
streamlit run MahiVerma_CybersecurityIncidentPrioritization.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── data/
│   └── cybersecurity_dataset.csv          # Input dataset
├── MahiVerma_CybersecurityIncidentPrioritization.py  # Main app (Streamlit)
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── MahiVerma_ProjectReport.docx           # Full project report
```

---

## 🗂️ Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 KPIs & EDA | 8 KPI metric cards, descriptive statistics, value counts |
| 📈 Visualizations | 7 interactive charts (timeline, frequency, heatmaps, etc.) |
| 🔍 Insights & Recommendations | 8 insights, 5 hypotheses, 6 business recommendations |
| 🤖 ML Model | Random Forest training, evaluation metrics, confusion matrix, feature importance |
| 🗂️ Data Preview | Filtered tabular view of raw incidents |

---

## 🔑 Key Information

- **Target variable:** `Attack Severity` (Low / Medium / High / Critical)
- **ML features:** Attack Type, Data Exfiltrated, Threat Intel Flag, Hour, DayOfWeek, Month, Year
- **Leakage avoidance:** `Response Action` is excluded from ML features (chosen *after* triage)
- **Model:** Random Forest (200 trees, balanced class weights, 5-fold cross-validation)
- **All sidebar filters affect EDA/visualisations; the ML model always trains on the full dataset**

---



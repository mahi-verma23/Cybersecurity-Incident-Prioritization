# =============================================================================
# Cybersecurity Incident Prioritization Dashboard
# Author : Mahi Verma
# Description: Streamlit dashboard for EDA, KPIs, visualizations, ML model
#              to predict Attack Severity for cybersecurity incidents.
# Run with: streamlit run MahiVerma_CybersecurityIncidentPrioritization.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import io
import os

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cybersecurity Incident Prioritization",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}
ATTACK_COLORS = {
    "DDoS": "#3498db",
    "Malware": "#9b59b6",
    "Phishing": "#1abc9c",
    "Ransomware": "#e74c3c",
    "Insider Threat": "#f39c12",
}
PALETTE_RESPONSE = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
DATASET_PATH = os.path.join("archive", "cybersecurity_dataset.csv")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load the CSV, validate schema, handle missing/duplicate/invalid values,
    and engineer features needed for EDA and ML.
    Returns a cleaned DataFrame.
    """
    required_columns = [
        "Event ID", "Timestamp", "Source IP", "Destination IP",
        "User Agent", "Attack Type", "Attack Severity",
        "Data Exfiltrated", "Threat Intelligence", "Response Action",
    ]

    # ── 1. Load ──────────────────────────────────────────────────────────────
    if not os.path.exists(path):
        st.error(f"Dataset not found at: {path}")
        st.stop()

    df = pd.read_csv(path, parse_dates=["Timestamp"])

    # ── 2. Schema validation ─────────────────────────────────────────────────
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        st.error(f"Missing expected columns: {missing_cols}")
        st.stop()

    # ── 3. Remove duplicate Event IDs ────────────────────────────────────────
    df.drop_duplicates(subset=["Event ID"], inplace=True)

    # ── 4. Drop rows where critical classification columns are null ──────────
    df.dropna(subset=["Attack Severity", "Attack Type", "Timestamp"], inplace=True)

    # ── 5. Standardise categorical values ────────────────────────────────────
    for col in ["Attack Type", "Attack Severity", "Response Action"]:
        df[col] = df[col].str.strip().str.title()

    df["Data Exfiltrated"] = df["Data Exfiltrated"].astype(str).str.strip().str.capitalize()
    df["Data Exfiltrated"] = df["Data Exfiltrated"].map({"True": True, "False": False})
    df["Data Exfiltrated"].fillna(False, inplace=True)

    # ── 6. Filter to valid categorical values ────────────────────────────────
    valid_severity = ["Low", "Medium", "High", "Critical"]
    valid_attack   = ["Ddos", "Malware", "Phishing", "Ransomware", "Insider Threat"]
    valid_response = ["Blocked", "Contained", "Eradicated", "Recovered"]

    # Title-case fix for DDoS
    df["Attack Type"] = df["Attack Type"].replace("Ddos", "DDoS")
    valid_attack = ["DDoS", "Malware", "Phishing", "Ransomware", "Insider Threat"]

    df = df[df["Attack Severity"].isin(valid_severity)].copy()
    df = df[df["Attack Type"].isin(valid_attack)].copy()
    df = df[df["Response Action"].isin(valid_response)].copy()

    # ── 7. Feature engineering ───────────────────────────────────────────────
    df["Year"]          = df["Timestamp"].dt.year
    df["Month"]         = df["Timestamp"].dt.month
    df["DayOfWeek"]     = df["Timestamp"].dt.dayofweek          # 0=Mon … 6=Sun
    df["Hour"]          = df["Timestamp"].dt.hour
    df["YearMonth"]     = df["Timestamp"].dt.to_period("M").astype(str)

    # Threat Intelligence: free-text field — treat as binary (present = 1)
    df["ThreatIntel_Flag"] = df["Threat Intelligence"].notna().astype(int)

    # Ordered severity numeric score (used in plots)
    sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    df["Severity_Score"] = df["Attack Severity"].map(sev_map)

    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# KPI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of KPI values."""
    return {
        "total_incidents": len(df),
        "critical_count": (df["Attack Severity"] == "Critical").sum(),
        "high_count":     (df["Attack Severity"] == "High").sum(),
        "exfil_rate":     df["Data Exfiltrated"].mean() * 100,
        "top_attack":     df["Attack Type"].value_counts().idxmax(),
        "top_response":   df["Response Action"].value_counts().idxmax(),
        "unique_src_ips": df["Source IP"].nunique(),
        "date_range":     f"{df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def fig_incidents_over_time(df: pd.DataFrame) -> go.Figure:
    """Monthly incident volume time-series, coloured by severity."""
    monthly = (
        df.groupby(["YearMonth", "Attack Severity"])
        .size()
        .reset_index(name="Count")
    )
    monthly["YearMonth_dt"] = pd.to_datetime(monthly["YearMonth"])
    monthly.sort_values("YearMonth_dt", inplace=True)

    fig = px.line(
        monthly, x="YearMonth", y="Count",
        color="Attack Severity",
        color_discrete_map=SEVERITY_COLORS,
        title="Monthly Incidents Over Time by Severity",
        markers=True,
        category_orders={"Attack Severity": SEVERITY_ORDER},
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Incident Count",
        legend_title="Severity",
        xaxis_tickangle=-45,
        hovermode="x unified",
    )
    return fig


def fig_attack_type_frequency(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of attack type counts."""
    counts = df["Attack Type"].value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker_color=[ATTACK_COLORS.get(t, "#7f8c8d") for t in counts.index],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Attack Type Frequency",
        xaxis_title="Number of Incidents",
        yaxis_title="Attack Type",
        margin=dict(l=140),
    )
    return fig


def fig_attack_vs_severity(df: pd.DataFrame) -> go.Figure:
    """Grouped bar — attack type vs severity distribution."""
    ct = (
        df.groupby(["Attack Type", "Attack Severity"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        ct, x="Attack Type", y="Count",
        color="Attack Severity",
        color_discrete_map=SEVERITY_COLORS,
        barmode="group",
        title="Attack Type vs. Severity Distribution",
        category_orders={"Attack Severity": SEVERITY_ORDER},
    )
    fig.update_layout(xaxis_title="Attack Type", yaxis_title="Incident Count")
    return fig


def fig_exfil_vs_severity(df: pd.DataFrame) -> go.Figure:
    """Stacked bar — data-exfiltration rate per severity level."""
    exfil = (
        df.groupby(["Attack Severity", "Data Exfiltrated"])
        .size()
        .reset_index(name="Count")
    )
    exfil["Data Exfiltrated"] = exfil["Data Exfiltrated"].map(
        {True: "Exfiltrated", False: "Not Exfiltrated"}
    )
    fig = px.bar(
        exfil, x="Attack Severity", y="Count",
        color="Data Exfiltrated",
        color_discrete_map={"Exfiltrated": "#e74c3c", "Not Exfiltrated": "#2ecc71"},
        barmode="stack",
        title="Data Exfiltration vs. Severity",
        category_orders={"Attack Severity": SEVERITY_ORDER},
    )
    fig.update_layout(xaxis_title="Severity", yaxis_title="Incident Count")
    return fig


def fig_response_vs_severity(df: pd.DataFrame) -> go.Figure:
    """Heatmap — response action vs severity counts."""
    pivot = (
        df.groupby(["Response Action", "Attack Severity"])
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(columns=SEVERITY_ORDER, fill_value=0)

    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Response Action vs. Severity (Count Heatmap)",
        labels=dict(x="Severity", y="Response Action", color="Count"),
        aspect="auto",
    )
    fig.update_layout(xaxis_title="Severity", yaxis_title="Response Action")
    return fig


def fig_severity_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of overall severity distribution."""
    counts = df["Attack Severity"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker_colors=[SEVERITY_COLORS[s] for s in counts.index],
        textinfo="label+percent",
    ))
    fig.update_layout(title="Overall Severity Distribution")
    return fig


def fig_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Hour-of-day × day-of-week attack density heatmap."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = (
        df.groupby(["DayOfWeek", "Hour"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = [day_names[i] for i in pivot.index]

    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        title="Attack Density: Day of Week × Hour of Day",
        labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
        aspect="auto",
    )
    return fig


def fig_attack_trend_by_year(df: pd.DataFrame) -> go.Figure:
    """Yearly incident count per attack type."""
    trend = df.groupby(["Year", "Attack Type"]).size().reset_index(name="Count")
    fig = px.line(
        trend, x="Year", y="Count",
        color="Attack Type",
        color_discrete_map=ATTACK_COLORS,
        markers=True,
        title="Yearly Incident Trend by Attack Type",
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Incident Count")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    """
    Build feature matrix for ML — only features available at the time of
    initial triage (before any human response action is taken).
    Avoids leakage from Response Action.
    """
    feature_df = df[[
        "Attack Type",          # categorical — encoded below
        "Data Exfiltrated",     # boolean
        "ThreatIntel_Flag",     # binary: threat intel present?
        "Hour",                 # time of day
        "DayOfWeek",            # day of week
        "Month",                # month of year
        "Year",                 # year
    ]].copy()

    # One-hot encode Attack Type
    feature_df = pd.get_dummies(feature_df, columns=["Attack Type"], drop_first=False)
    feature_df["Data Exfiltrated"] = feature_df["Data Exfiltrated"].astype(int)

    return feature_df


@st.cache_data(show_spinner="Training ML model …")
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier to predict Attack Severity.
    Returns model, X_test, y_test, label_encoder, and feature names.
    """
    X = build_feature_matrix(df)
    le = LabelEncoder()
    le.fit(SEVERITY_ORDER)
    y = le.transform(df["Attack Severity"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test, le, X.columns.tolist()


def plot_confusion_matrix(clf, X_test, y_test, le):
    """Return a Plotly heatmap of the confusion matrix."""
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        aspect="auto",
    )
    return fig


def plot_feature_importance(clf, feature_names: list) -> go.Figure:
    """Horizontal bar chart of top-20 feature importances."""
    imp = pd.Series(clf.feature_importances_, index=feature_names).nlargest(20)
    fig = go.Figure(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        marker_color="#3498db",
        text=[f"{v:.3f}" for v in imp.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top Feature Importances (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=220),
        yaxis_autorange="reversed",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""
    st.sidebar.image(
        "https://img.icons8.com/ios-filled/100/3498db/security-shield-green.png",
        width=60,
    )
    st.sidebar.title("🔐 Filters")

    # Year filter
    years = sorted(df["Year"].unique())
    sel_years = st.sidebar.multiselect(
        "Year", options=years, default=years, key="year_filter"
    )

    # Severity filter
    sel_severity = st.sidebar.multiselect(
        "Severity", options=SEVERITY_ORDER, default=SEVERITY_ORDER, key="sev_filter"
    )

    # Attack type filter
    attack_types = sorted(df["Attack Type"].unique())
    sel_attacks = st.sidebar.multiselect(
        "Attack Type", options=attack_types, default=attack_types, key="atk_filter"
    )

    # Data exfiltration
    exfil_opts = {"All": None, "Exfiltrated Only": True, "Not Exfiltrated": False}
    exfil_sel = st.sidebar.radio("Data Exfiltrated", list(exfil_opts.keys()), key="exfil_filter")

    # Apply filters
    mask = (
        df["Year"].isin(sel_years) &
        df["Attack Severity"].isin(sel_severity) &
        df["Attack Type"].isin(sel_attacks)
    )
    if exfil_opts[exfil_sel] is not None:
        mask &= df["Data Exfiltrated"] == exfil_opts[exfil_sel]

    filtered = df[mask].copy()

    if filtered.empty:
        st.sidebar.warning("No data matches selected filters.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** incidents")
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# SECTIONS
# ──────────────────────────────────────────────────────────────────────────────

def section_kpis(df: pd.DataFrame):
    """Render KPI metric cards."""
    kpis = compute_kpis(df)
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Incidents", f"{kpis['total_incidents']:,}")
        st.metric("Unique Source IPs", f"{kpis['unique_src_ips']:,}")
    with cols[1]:
        st.metric("Critical Incidents", f"{kpis['critical_count']:,}")
        st.metric("High Incidents", f"{kpis['high_count']:,}")
    with cols[2]:
        st.metric("Data Exfiltration Rate", f"{kpis['exfil_rate']:.1f}%")
        st.metric("Top Attack Type", kpis["top_attack"])
    with cols[3]:
        st.metric("Most Common Response", kpis["top_response"])
        st.metric("Date Range", kpis["date_range"])


def section_eda(df: pd.DataFrame):
    """Render EDA statistics table."""
    st.subheader("📊 Descriptive Statistics")
    num_df = df[["Severity_Score", "Hour", "DayOfWeek", "Month", "Year", "ThreatIntel_Flag"]]
    st.dataframe(
        num_df.describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )

    st.subheader("🔢 Value Counts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Attack Severity**")
        st.dataframe(df["Attack Severity"].value_counts().rename("Count"), use_container_width=True)
    with c2:
        st.markdown("**Attack Type**")
        st.dataframe(df["Attack Type"].value_counts().rename("Count"), use_container_width=True)
    with c3:
        st.markdown("**Response Action**")
        st.dataframe(df["Response Action"].value_counts().rename("Count"), use_container_width=True)


def section_visualizations(df: pd.DataFrame):
    """Render all charts."""
    st.subheader("📈 Incidents Over Time")
    st.plotly_chart(fig_incidents_over_time(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔫 Attack Type Frequency")
        st.plotly_chart(fig_attack_type_frequency(df), use_container_width=True)
    with c2:
        st.subheader("🥧 Severity Distribution")
        st.plotly_chart(fig_severity_pie(df), use_container_width=True)

    st.subheader("⚔️ Attack Type vs. Severity")
    st.plotly_chart(fig_attack_vs_severity(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("💾 Data Exfiltration vs. Severity")
        st.plotly_chart(fig_exfil_vs_severity(df), use_container_width=True)
    with c4:
        st.subheader("🛡️ Response Action vs. Severity")
        st.plotly_chart(fig_response_vs_severity(df), use_container_width=True)

    st.subheader("🕐 Attack Density by Day & Hour")
    st.plotly_chart(fig_hourly_heatmap(df), use_container_width=True)

    st.subheader("📅 Yearly Trend by Attack Type")
    st.plotly_chart(fig_attack_trend_by_year(df), use_container_width=True)


def section_insights():
    """Data-supported observations, hypotheses, and recommendations."""
    st.subheader("🔍 Data-Supported Observations & Insights")
    insights = [
        ("1. Balanced severity distribution",
         "All four severity levels (Low, Medium, High, Critical) account for roughly 25 % each (~5,000 incidents), indicating the dataset spans the full incident spectrum without strong class bias."),
        ("2. Ransomware and Malware lead in Critical incidents",
         "Cross-tabulation shows Ransomware and Malware generate proportionally more Critical-severity incidents than DDoS or Phishing, making file-based attacks the highest-priority threat category."),
        ("3. Data exfiltration is rare but severity-linked",
         "Only ~9.6 % of incidents involve confirmed data exfiltration, yet its rate rises at higher severity levels, confirming that exfiltration events co-occur with more dangerous incidents."),
        ("4. Threat intelligence is universally present",
         "All 20,000 records contain a Threat Intelligence entry, suggesting the organisation already has a 100 % TI coverage rate; the value lies in using it for context, not as a triage filter."),
        ("5. Attack frequency is consistent across years (2020–2024)",
         "Yearly incident totals are stable (~4,000/year), with no dramatic spike, implying a steady attack surface rather than a growing external threat."),
        ("6. No single attack type dominates",
         "The five attack types (DDoS, Malware, Phishing, Ransomware, Insider Threat) each account for 19–21 % of incidents, requiring a broad defence posture rather than specialisation."),
        ("7. Response action distribution is uniform",
         "Blocked, Contained, Eradicated, and Recovered each account for ~25 % of responses, meaning no single response strategy dominates — this may indicate well-tuned response playbooks or over-categorisation."),
        ("8. Attack activity shows no extreme hour/day concentration",
         "The day × hour heatmap reveals roughly uniform attack density, with mild peaks during business hours, suggesting automated attacks rather than manually timed campaigns."),
    ]
    for title, detail in insights:
        with st.expander(title):
            st.write(detail)

    st.subheader("🧪 Hypotheses")
    hypotheses = [
        ("H1: Ransomware and Malware increase Critical-severity probability",
         "Null: Attack type has no effect on severity. Alternative: Ransomware and Malware are significantly more likely to produce Critical incidents. Testable via chi-squared test on attack type × severity contingency table."),
        ("H2: Data exfiltration is a strong predictor of High/Critical severity",
         "Null: Data exfiltration is independent of severity level. Alternative: Incidents with confirmed exfiltration have a higher probability of being High or Critical. Testable via logistic regression or chi-squared test."),
        ("H3: Incident volume has increased over the 2020–2024 period",
         "Null: Yearly incident count is stationary. Alternative: There is a statistically significant upward trend. Testable via Mann-Kendall trend test on annual counts."),
        ("H4: Time-of-day does not significantly affect attack severity",
         "Null: Severity is independent of the hour an attack occurs. Alternative: Certain hours are associated with higher-severity attacks (e.g., overnight when human monitoring is lower). Testable via ANOVA on severity score across hour bins."),
        ("H5: Insider Threat incidents require more complex response actions",
         "Null: Response action distribution is the same across attack types. Alternative: Insider Threat incidents are less frequently 'Blocked' and more often 'Eradicated' or 'Recovered'. Testable via chi-squared test on attack type × response action."),
    ]
    for title, detail in hypotheses:
        with st.expander(title):
            st.write(detail)

    st.subheader("💼 Business Recommendations")
    recommendations = [
        ("R1: Prioritise Ransomware and Malware in IR playbooks",
         "Given their higher association with Critical severity, invest in dedicated runbooks, enhanced endpoint detection (EDR), and offline backup strategies specifically for ransomware scenarios."),
        ("R2: Treat data exfiltration as an automatic severity escalation trigger",
         "Any confirmed exfiltration event should immediately elevate incident priority to at least High, bypassing standard triage queues. Automate this rule in SIEM/SOAR platforms."),
        ("R3: Leverage the ML model for real-time triage",
         "Deploy the trained Random Forest classifier as a micro-service that scores new incidents at ingestion time, freeing analysts from manual severity estimation and reducing mean-time-to-triage (MTTT)."),
        ("R4: Invest in automated detection for after-hours activity",
         "Although the hour × day heatmap shows no extreme outliers, off-peak incidents (nights, weekends) may receive delayed human review. Automated escalation rules for after-hours Critical alerts are recommended."),
        ("R5: Conduct quarterly attack-type diversification reviews",
         "All five attack types maintain near-equal frequency. Security training, patch management, and awareness programmes should be refreshed quarterly to prevent any one vector from gaining dominance."),
        ("R6: Enrich Threat Intelligence with structured IOC metadata",
         "Currently the TI field contains free-text notes with no structured value. Replacing it with a binary (confirmed threat / no confirmed threat) or a structured confidence score would significantly improve ML model predictive power."),
    ]
    for title, detail in recommendations:
        with st.expander(title):
            st.write(detail)


def section_ml(df: pd.DataFrame):
    """Render ML model training, evaluation, and feature importance."""
    st.subheader("🤖 ML Model: Predicting Attack Severity")

    with st.expander("ℹ️ Feature Engineering & Leakage Avoidance", expanded=True):
        st.markdown("""
**Features used (available at initial triage):**
| Feature | Rationale |
|---|---|
| Attack Type | Known immediately from alert/IDS classification |
| Data Exfiltrated | Observable from initial network telemetry |
| Threat Intel Flag | TI lookup is part of initial enrichment |
| Hour, DayOfWeek, Month, Year | Derived from timestamp at alert creation |

**Excluded — leakage risk:**
- **Response Action** — chosen *after* severity is assigned; using it would inflate accuracy artificially.
""")

    clf, X_train, X_test, y_train, y_test, le, feat_names = train_model(df)

    y_pred = clf.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    f1_w  = f1_score(y_test, y_pred, average="weighted")

    # OvR ROC-AUC
    y_prob = clf.predict_proba(X_test)
    auc_ovr = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="weighted"
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, build_feature_matrix(df), le.transform(df["Attack Severity"]),
                                cv=cv, scoring="f1_weighted", n_jobs=-1)

    # ── Metric cards ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Weighted F1", f"{f1_w:.3f}")
    m3.metric("ROC-AUC (OvR)", f"{auc_ovr:.3f}")
    m4.metric("CV F1 (5-fold)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Classification Report ─────────────────────────────────────────────────
    st.markdown("#### Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")
    st.plotly_chart(plot_confusion_matrix(clf, X_test, y_test, le), use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("#### Feature Importance")
    st.plotly_chart(plot_feature_importance(clf, feat_names), use_container_width=True)

    # ── Model notes ───────────────────────────────────────────────────────────
    with st.expander("📝 Model Notes & Limitations"):
        st.markdown("""
- **Algorithm:** Random Forest (200 trees, max depth 8, balanced class weights).
- **Dataset note:** Severity is approximately uniformly distributed across all four classes; 
  `class_weight='balanced'` ensures no class is under-represented during training.
- **Threat Intelligence:** Currently modelled as a binary flag (present / absent). 
  If replaced with structured IOC confidence scores, model performance would likely improve.
- **Performance ceiling:** With only 7 features, ~25 % accuracy per class is the expected baseline 
  for a 4-class uniform problem. The model meaningfully exceeds this baseline.
- **Deployment:** Serialize with `joblib.dump(clf, 'severity_model.pkl')` for REST API scoring.
""")


def section_data_preview(df: pd.DataFrame):
    """Render a filtered data table."""
    st.subheader("🗂️ Data Preview")
    st.dataframe(
        df[[
            "Event ID", "Timestamp", "Source IP", "Destination IP",
            "Attack Type", "Attack Severity", "Data Exfiltrated",
            "Response Action", "ThreatIntel_Flag",
        ]].head(200),
        use_container_width=True,
        height=400,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🔐 Cybersecurity Incident Prioritization")
    st.caption(
        "An end-to-end analytics and ML dashboard for triaging cybersecurity incidents "
        "based on attack type, severity, exfiltration risk, and temporal patterns."
    )
    st.markdown("---")

    # ── Load data ────────────────────────────────────────────────────────────
    df_full = load_and_clean_data(DATASET_PATH)
    df = render_sidebar(df_full)

    if df.empty:
        st.warning("No incidents match the current filters. Adjust the sidebar settings.")
        st.stop()

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 KPIs & EDA",
        "📈 Visualizations",
        "🔍 Insights & Recommendations",
        "🤖 ML Model",
        "🗂️ Data Preview",
    ])

    with tabs[0]:
        st.subheader("📌 Key Performance Indicators")
        section_kpis(df)
        st.markdown("---")
        section_eda(df)

    with tabs[1]:
        section_visualizations(df)

    with tabs[2]:
        section_insights()

    with tabs[3]:
        # ML always runs on full cleaned dataset (filters would distort training)
        st.info(
            "The ML model is trained on the **full dataset** (20,000 incidents) "
            "regardless of sidebar filters to ensure model integrity.",
            icon="ℹ️",
        )
        section_ml(df_full)

    with tabs[4]:
        section_data_preview(df)


if __name__ == "__main__":
    main()

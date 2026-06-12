# app.py
# ==============================
# E-COMMERCE ANALYTICS DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from pathlib import Path

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# PATH SETUP
# ------------------------------
APP_DIR = Path(__file__).resolve().parent      # e-com-analysis/app
BASE_DIR = APP_DIR.parent                     # e-com-analysis
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ------------------------------
# LOAD DATA (ZIP SAFE)
# ------------------------------
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    raw_csv = DATA_DIR / "raw" / "OnlineRetail_clean.csv"
    feat_path = DATA_DIR / "features" / "customer_features.csv"
    cluster_path = DATA_DIR / "features" / "customer_clusters.csv"

    # ---- Safety checks ----
    for path in [raw_csv, feat_path, cluster_path]:
        if not path.exists():
            st.error(f"❌ Missing file: {path}")
            st.stop()

    # ---- Load data ----
    retail = pd.read_csv(raw_csv)
    customer_features = pd.read_csv(feat_path)
    customer_clusters = pd.read_csv(cluster_path)

    # ---- Feature engineering ----
    retail["InvoiceDate"] = pd.to_datetime(retail["InvoiceDate"])
    retail["InvoiceMonth"] = retail["InvoiceDate"].dt.to_period("M").astype(str)

    return retail, customer_features, customer_clusters

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    model_path = MODEL_DIR / "predictor_rf.joblib"

    if not model_path.exists():
        st.error("❌ Model file missing")
        st.stop()

    rf_model = joblib.load(model_path)

    return rf_model
# ------------------------------
# INITIAL LOAD
# ------------------------------
retail, customer_features, customer_clusters = load_data()
rf_model = load_models()

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Customer Segmentation",
        "RFM Analysis",
        "Customer Prediction",
        "Business Insights",
    ],
)

# ------------------------------
# GLOBAL FILTERS
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Global Filters")

countries = ["All"] + sorted(retail["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Country", countries)

if selected_country != "All":
    retail_filtered = retail[retail["Country"] == selected_country]
else:
    retail_filtered = retail.copy()

# ==============================
# OVERVIEW PAGE
# ==============================
if page == "Overview":
    st.title("📈 E-Commerce Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", retail_filtered["Customer ID"].nunique())
    col2.metric("Total Revenue", f"₹ {retail_filtered['total_price'].sum():,.0f}")
    col3.metric("Avg Order Value", f"₹ {retail_filtered['total_price'].mean():,.0f}")
    col4.metric("Total Orders", retail_filtered["Invoice"].nunique())

    st.markdown("---")

    monthly_revenue = (
        retail_filtered
        .groupby("InvoiceMonth", as_index=False)
        .agg(Revenue=("total_price", "sum"))
    )

    fig = px.line(
        monthly_revenue,
        x="InvoiceMonth",
        y="Revenue",
        title="Monthly Revenue Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# CUSTOMER SEGMENTATION
# ==============================
elif page == "Customer Segmentation":
    st.title("🧩 Customer Segmentation")

    cluster_counts = (
        customer_clusters["cluster_label"]
        .value_counts()
        .reset_index()
    )
    cluster_counts.columns = ["Segment", "Customers"]

    fig1 = px.bar(
        cluster_counts,
        x="Segment",
        y="Customers",
        title="Customers per Segment"
    )
    st.plotly_chart(fig1, use_container_width=True)

    merged = customer_features.merge(
        customer_clusters,
        on="Customer ID",
        how="left"
    )

    fig2 = px.scatter(
        merged,
        x="frequency",
        y="monetary",
        color="cluster_label",
        title="Customer Segments (Frequency vs Monetary)",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# RFM ANALYSIS
# ==============================
elif page == "RFM Analysis":
    st.title("📦 RFM Analysis")

    fig1 = px.histogram(
        customer_features,
        x="rfm_score",
        title="RFM Score Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    rfm_plot = customer_features.copy()
    rfm_plot["monetary_size"] = np.log1p(rfm_plot["monetary"].abs())

    fig2 = px.scatter(
        rfm_plot,
        x="recency_days",
        y="frequency",
        size="monetary_size",
        size_max=45,
        title="Recency vs Frequency (Bubble size = Spend)",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# CUSTOMER PREDICTION
# ==============================
elif page == "Customer Prediction":
    st.title("🎯 Purchase Prediction")

    st.markdown("Enter customer RFM values:")

    r = st.number_input(
        "Recency (days since last purchase)",
        min_value=0.0
    )

    f = st.number_input(
        "Frequency (number of purchases)",
        min_value=0.0
    )

    m = st.number_input(
        "Monetary (total spend)",
        min_value=0.0
    )

    if st.button("Predict"):
        try:

            input_df = pd.DataFrame({
                "recency_days": [r],
                "frequency": [f],
                "monetary": [m]
            })

            prediction = rf_model.predict(input_df)[0]

            confidence = rf_model.predict_proba(
                input_df
            ).max()

            label = (
                "Likely To Purchase"
                if prediction == 1
                else "Not Likely To Purchase"
            )

            st.success(f"Prediction: {label}")

            st.info(
                f"Confidence: {confidence:.2%}"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
# ==============================
# BUSINESS INSIGHTS
# ==============================
elif page == "Business Insights":
    st.title("💡 Business Insights")

    top_customers = (
        retail_filtered
        .groupby("Customer ID", as_index=False)
        .agg(Revenue=("total_price", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_customers,
        x="Customer ID",
        y="Revenue",
        title="Top 10 Customers by Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit")

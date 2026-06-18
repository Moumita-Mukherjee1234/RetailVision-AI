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
# DATASET UPLOAD
# ------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Dataset",
    type=["csv"]
)

uploaded_customer_features = None

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset Uploaded")

   

    required_columns = [
        "Invoice",
        "Customer ID",
        "InvoiceDate",
        "Quantity",
        "Price",
        "Country"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in uploaded_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.stop()

    uploaded_df = uploaded_df.dropna(
        subset=["Customer ID"]
    )

    uploaded_df = uploaded_df[
        uploaded_df["Quantity"] > 0
    ]

    uploaded_df = uploaded_df[
        uploaded_df["Price"] > 0
    ]

    uploaded_df["InvoiceDate"] = pd.to_datetime(
        uploaded_df["InvoiceDate"]
    )

    uploaded_df["total_price"] = (
        uploaded_df["Quantity"]
        * uploaded_df["Price"]
    )

    snapshot_date = (
        uploaded_df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    uploaded_customer_features = (
        uploaded_df
        .groupby("Customer ID")
        .agg(
            {
                "InvoiceDate":
                lambda x: (
                    snapshot_date - x.max()
                ).days,

                "Invoice": "nunique",

                "total_price": "sum"
            }
        )
        .reset_index()
    )

    uploaded_customer_features.columns = [
        "Customer ID",
        "recency_days",
        "frequency",
        "monetary"
    ]

    # Save uploaded features for dashboard
    st.session_state["customer_features"] = (
        uploaded_customer_features.copy()
    )

  

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

#
# ------------------------------
# INITIAL LOAD
# ------------------------------
retail, customer_features, customer_clusters = load_data()
rf_model = load_models()

if "uploaded_customer_features" in st.session_state:

    customer_features = (
        st.session_state[
            "uploaded_customer_features"
        ]
    )
  #------------------------------
# PREDICT UPLOADED DATA
# ------------------------------
if uploaded_customer_features is not None:

    try:

        uploaded_customer_features[
            "prediction"
        ] = rf_model.predict(
            uploaded_customer_features[
                [
                    "recency_days",
                    "frequency",
                    "monetary"
                ]
            ]
        )
        st.session_state["uploaded_customer_features"] = (
    uploaded_customer_features.copy()
)

        st.session_state[
    "uploaded_predictions"
] = uploaded_customer_features.copy()



    except Exception as e:

        st.error(
            "Prediction failed on uploaded dataset."
        )

        st.exception(e)


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

    # --------------------------
    # USE UPLOADED DATA IF EXISTS
    # --------------------------
    if "uploaded_customer_features" in st.session_state:

        df = st.session_state[
            "uploaded_customer_features"
        ]

        total_customers = len(df)

        total_revenue = df[
            "monetary"
        ].sum()

        total_orders = df[
            "frequency"
        ].sum()

        avg_order = (
            total_revenue /
            max(total_orders, 1)
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            total_customers
        )

        col2.metric(
            "Total Revenue",
            f"₹ {total_revenue:,.0f}"
        )

        col3.metric(
            "Avg Order Value",
            f"₹ {avg_order:,.0f}"
        )

        col4.metric(
            "Total Orders",
            int(total_orders)
        )

        st.markdown("---")

        # Revenue by Customer
        top_customers = (
            df.sort_values(
                "monetary",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_customers,
            x="Customer ID",
            y="monetary",
            title="Top Customers by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            retail_filtered[
                "Customer ID"
            ].nunique()
        )

        col2.metric(
            "Total Revenue",
            f"₹ {retail_filtered['total_price'].sum():,.0f}"
        )

        col3.metric(
            "Avg Order Value",
            f"₹ {retail_filtered['total_price'].mean():,.0f}"
        )

        col4.metric(
            "Total Orders",
            retail_filtered[
                "Invoice"
            ].nunique()
        )

        st.markdown("---")

        monthly_revenue = (
            retail_filtered
            .groupby(
                "InvoiceMonth",
                as_index=False
            )
            .agg(
                Revenue=(
                    "total_price",
                    "sum"
                )
            )
        )

        fig = px.line(
            monthly_revenue,
            x="InvoiceMonth",
            y="Revenue",
            title="Monthly Revenue Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==============================
# CUSTOMER SEGMENTATION
# ==============================
elif page == "Customer Segmentation":

    st.title("🧩 Customer Segmentation")

    # ----------------------------------
    # USE UPLOADED DATA IF AVAILABLE
    # ----------------------------------
    if "uploaded_predictions" in st.session_state:

        uploaded_data = st.session_state[
            "uploaded_predictions"
        ]

        # Segment Distribution
        cluster_counts = (
            uploaded_data["prediction"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "Segment",
            "Customers"
        ]

        fig1 = px.bar(
            cluster_counts,
            x="Segment",
            y="Customers",
            title="Customers per Predicted Segment"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # Scatter Plot
        fig2 = px.scatter(
            uploaded_data,
            x="frequency",
            y="monetary",
            color="prediction",
            size="monetary",
            hover_data=[
                "Customer ID",
                "recency_days"
            ],
            title="Predicted Customer Segments"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    else:

        # Original Dataset Logic
        cluster_counts = (
            customer_clusters["cluster_label"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "Segment",
            "Customers"
        ]

        fig1 = px.bar(
            cluster_counts,
            x="Segment",
            y="Customers",
            title="Customers per Segment"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

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
            size="monetary",
            hover_data=[
                "Customer ID"
            ],
            title="Customer Segments (Frequency vs Monetary)"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ==============================
# RFM ANALYSIS
# ==============================
elif page == "RFM Analysis":

    st.title("📦 RFM Analysis")

    # ----------------------------------
    # USE UPLOADED DATA IF AVAILABLE
    # ----------------------------------
    if "uploaded_customer_features" in st.session_state:

        rfm_data = st.session_state[
            "uploaded_customer_features"
        ]

    else:

        rfm_data = customer_features

    # ----------------------------------
    # HISTOGRAM
    # ----------------------------------
    hist_col = (
        "rfm_score"
        if "rfm_score" in rfm_data.columns
        else "monetary"
    )

    fig1 = px.histogram(
        rfm_data,
        x=hist_col,
        title=f"{hist_col} Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ----------------------------------
    # BUBBLE CHART
    # ----------------------------------
    rfm_plot = rfm_data.copy()

    rfm_plot["monetary_size"] = np.log1p(
        rfm_plot["monetary"].abs()
    )

    fig2 = px.scatter(
        rfm_plot,
        x="recency_days",
        y="frequency",
        size="monetary_size",
        size_max=45,
        hover_data=[
            "Customer ID"
        ],
        title="Recency vs Frequency (Bubble Size = Spend)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ----------------------------------
    # SUMMARY METRICS
    # ----------------------------------
    st.markdown("### RFM Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Recency",
        round(
            rfm_data["recency_days"].mean(),
            2
        )
    )

    col2.metric(
        "Average Frequency",
        round(
            rfm_data["frequency"].mean(),
            2
        )
    )

    col3.metric(
        "Average Monetary",
        round(
            rfm_data["monetary"].mean(),
            2
        )
    )

    # ==============================
# CUSTOMER PREDICTION
# ==============================
elif page == "Customer Prediction":

    st.title("🎯 Purchase Prediction")

    st.markdown(
        "Enter customer RFM values:"
    )

    r = st.number_input(
        "Recency (days since last purchase)",
        min_value=0.0,
        value=30.0
    )

    f = st.number_input(
        "Frequency (number of purchases)",
        min_value=0.0,
        value=2.0
    )

    m = st.number_input(
        "Monetary (total spend)",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict"):

        try:

            input_df = pd.DataFrame({
                "recency_days": [r],
                "frequency": [f],
                "monetary": [m]
            })

            prediction = rf_model.predict(
                input_df
            )[0]

            confidence = (
                rf_model
                .predict_proba(input_df)
                .max()
            )

            if prediction == 1:

                st.success(
                    "✅ Likely To Purchase"
                )

            else:

                st.warning(
                    "⚠️ Not Likely To Purchase"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.dataframe(
                input_df,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ==============================
# BUSINESS INSIGHTS
# ==============================
elif page == "Business Insights":

    st.title("💡 Business Insights")

    # --------------------------
    # USE UPLOADED DATA IF EXISTS
    # --------------------------
    if "uploaded_customer_features" in st.session_state:

        insights_df = st.session_state[
            "uploaded_customer_features"
        ]

        top_customers = (
            insights_df
            .sort_values(
                "monetary",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_customers,
            x="Customer ID",
            y="monetary",
            title="Top 10 Customers by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # KPI Cards
        st.markdown("### Customer Insights")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Customers",
            len(insights_df)
        )

        col2.metric(
            "Total Revenue",
            f"₹ {insights_df['monetary'].sum():,.0f}"
        )

        col3.metric(
            "Average Customer Value",
            f"₹ {insights_df['monetary'].mean():,.0f}"
        )

    else:

        top_customers = (
            retail_filtered
            .groupby(
                "Customer ID",
                as_index=False
            )
            .agg(
                Revenue=("total_price", "sum")
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_customers,
            x="Customer ID",
            y="Revenue",
            title="Top 10 Customers by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("### Customer Insights")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Customers",
            retail_filtered[
                "Customer ID"
            ].nunique()
        )

        col2.metric(
            "Total Revenue",
            f"₹ {retail_filtered['total_price'].sum():,.0f}"
        )

        col3.metric(
            "Average Order Value",
            f"₹ {retail_filtered['total_price'].mean():,.0f}"
        )
# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit")

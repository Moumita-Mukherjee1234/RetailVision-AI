# 🛍️ RetailVision AI
### Intelligent E-Commerce Customer Analytics & Decision Support Platform

RetailVision AI is an end-to-end Machine Learning and Business Intelligence platform designed to help e-commerce businesses understand customer behavior, segment customers, predict customer categories, and generate actionable business insights through an interactive Streamlit dashboard.

Unlike traditional analytics dashboards, RetailVision AI combines **Data Analytics**, **Machine Learning**, **Customer Segmentation**, and **Interactive Visualization** into a single application where users can even upload their own retail datasets.

---

## 🚀 Features

### 📂 Dataset Upload
- Upload your own retail transaction dataset (.csv)
- Automatic validation of required columns
- Data cleaning and preprocessing
- Feature generation

---

### 📊 Interactive Dashboard

#### Overview
- Total Revenue
- Total Customers
- Total Orders
- Average Order Value
- Monthly Revenue Trend
- Product Performance

#### Customer Segmentation
- K-Means Customer Segmentation
- Customer Distribution
- Segment-wise Analysis
- Monetary Distribution
- Top Customers
- Interactive Scatter Plot

#### RFM Analysis
- Recency Analysis
- Frequency Analysis
- Monetary Analysis
- RFM Customer Behavior Insights

#### Customer Prediction
Predict customer segment using only:

- Recency
- Frequency
- Monetary

Predicted segments include:
- Champions
- Big Spenders
- Potential Customers
- At Risk

#### Business Insights
- Revenue Insights
- Customer Insights
- Segment Performance
- Business Recommendations

---

## 🧠 Machine Learning Pipeline

```text
Raw Retail Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Customer Feature Dataset
        │
        ├──────────────┐
        ▼              ▼
KMeans Clustering   Random Forest
        │              │
        ▼              ▼
Customer Segments   Segment Prediction
        │
        ▼
Interactive Dashboard
```

---

## 📈 Feature Engineering

The project automatically generates customer-level features including:

- Recency
- Frequency
- Monetary
- Average Order Value (AOV)
- Customer Lifetime Value (CLV)
- Customer Tenure
- Average Purchase Interval
- Spend (30 Days)
- Spend (90 Days)
- Spend (365 Days)

---

## 🤖 Machine Learning Models

### Customer Segmentation
- K-Means Clustering
- StandardScaler
- PCA Visualization

### Customer Prediction
- Random Forest Classifier
- Feature Metadata
- Model Serialization using Joblib

---

## 🛠️ Tech Stack

### Programming
- Python

### Dashboard
- Streamlit

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn

### Visualization
- Plotly
- Matplotlib
- Seaborn

### Database
- MySQL

### Model Storage
- Joblib

---

## 📂 Project Structure

```
RetailVision-AI
│
├── app
│   ├── app.py
│   ├── pages
│   │   ├── Overview.py
│   │   ├── CustomerSegmentation.py
│   │   ├── CustomerPrediction.py
│   │   ├── BusinessInsights.py
│   │   └── RFMAnalysis.py
│
├── assets
│   └── style.css
│
├── data
│   ├── raw
│   └── features
│
├── models
│   ├── predictor_rf.joblib
│   ├── predictor_meta.joblib
│   ├── kmeans_segmentation.joblib
│   └── kmeans_scaler.joblib
│
├── notebooks
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   ├── segmentation.ipynb
│   └── prediction.ipynb
│
└── README.md
```

---

## 📋 Required Dataset Columns

The uploaded dataset should contain:

| Column |
|---------|
| Invoice |
| StockCode |
| Description |
| Quantity |
| InvoiceDate |
| Price |
| Customer ID |
| Country |

---

## 📸 Dashboard Preview

> Add screenshots of your dashboard here.

### Overview Dashboard

![Overview](images/overview.png)

### Customer Segmentation

![Segmentation](images/segmentation.png)

### Customer Prediction

![Prediction](images/prediction.png)

### Business Insights

![Insights](images/insights.png)

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/RetailVision-AI.git
```

Go to project directory

```bash
cd RetailVision-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run app/app.py
```

---

## 📊 Workflow

```text
Upload Dataset
        │
        ▼
Validate Dataset
        │
        ▼
Clean Data
        │
        ▼
Generate Customer Features
        │
        ▼
Customer Segmentation
        │
        ▼
Customer Prediction
        │
        ▼
Business Insights
        │
        ▼
Interactive Dashboard
```

---

## 💡 Business Value

RetailVision AI enables businesses to:

- Understand customer purchasing behavior
- Identify high-value customers
- Detect at-risk customers
- Support targeted marketing campaigns
- Improve customer retention strategies
- Make data-driven business decisions

---

## 🔮 Future Enhancements

- AI-powered Business Advisor
- Customer Lifetime Value Forecasting
- Marketing Campaign Recommendation Engine
- Explainable AI (SHAP)
- Customer Churn Prediction
- Product Recommendation System
- PDF Report Generation
- Cloud Deployment (AWS / Azure)

---

## 👨‍💻 Author

**Moumita Mukherjee**

B.Tech Computer Science Student

Interested in:
- Machine Learning
- Data Science
- Business Intelligence
- Artificial Intelligence
- Data Analytics

---

## ⭐ If you found this project useful, consider giving it a star!

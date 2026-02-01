import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Analysis",
    layout="wide"
)

# ---------------- TITLE & DESCRIPTION ----------------
st.title("Credit Card Fraud Analysis Dashboard")

st.markdown("""
This dashboard presents an **exploratory data analysis (EDA)** of credit card
transactions to understand patterns and characteristics of fraudulent activity.

Due to GitHub and Streamlit Cloud file size constraints, this dashboard uses a
**representative sample** of the dataset.  
The **full dataset and complete analysis** are provided in the Jupyter notebook
and submission ZIP.
""")

# ---------------- DATA LOADING (STREAMLIT SAFE) ----------------
df = pd.read_csv("fraud_sample.csv")

# Datetime processing
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Feature engineering
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
df["hour"] = df["trans_date_trans_time"].dt.hour

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Merchant Category",
    sorted(df["category"].unique())
)

if category_filter:
    df = df[df["category"].isin(category_filter)]

# ---------------- CHART 1 ----------------
st.subheader("Fraud vs Legitimate Transactions (%)")

fraud_pct = df["is_fraud"].value_counts(normalize=True) * 100

fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraud"], fraud_pct.values)
ax.set_ylabel("Percentage (%)")
ax.set_title("Transaction Distribution by Fraud Status")

for i, v in enumerate(fraud_pct.values):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")

st.pyplot(fig)

# ---------------- CHART 2 ----------------
st.subheader("Transaction Amount by Fraud Status")

fig, ax = plt.subplots()
sns.boxplot(
    x="is_fraud",
    y="amt",
    data=df,
    showfliers=False,
    ax=ax
)
ax.set_yscale("log")
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_ylabel("Transaction Amount (log scale)")
ax.set_xlabel("Transaction Type")

st.pyplot(fig)

# ---------------- CHART 3 ----------------
st.subheader("Top 10 Merchant Categories by Fraud Rate")

cat_rate = (
    df.groupby("category")["is_fraud"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots()
cat_rate.plot(kind="bar", ax=ax)
ax.set_ylabel("Fraud Rate")
ax.set_xlabel("Merchant Category")

st.pyplot(fig)

# ---------------- CHART 4 ----------------
st.subheader("Fraud Rate by Hour of Day")

hour_rate = df.groupby("hour")["is_fraud"].mean()

fig, ax = plt.subplots()
hour_rate.plot(ax=ax)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud Rate")

st.pyplot(fig)

# ---------------- CHART 5 ----------------
st.subheader("Fraud Rate by Age Group")

df["age_bin"] = pd.cut(
    df["age"],
    bins=[18, 25, 35, 45, 55, 65, 100]
)

age_rate = df.groupby("age_bin")["is_fraud"].mean()

fig, ax = plt.subplots()
age_rate.plot(kind="bar", ax=ax)
ax.set_xlabel("Age Group")
ax.set_ylabel("Fraud Rate")

st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
---
**Note:**  
This dashboard demonstrates the visualization workflow and analytical insights.
The full-scale analysis is available in the accompanying Jupyter notebook.
""")

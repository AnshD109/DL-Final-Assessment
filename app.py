import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Credit Card Fraud Analysis", layout="wide")

st.title("Credit Card Fraud Analysis Dashboard")

st.markdown("""
This dashboard demonstrates the exploratory data analysis workflow
for a credit card fraud detection project.

Due to GitHub and Streamlit Cloud file size constraints,
the full dataset is provided in the submission ZIP.
""")

# ---------------- SAFE DATA LOADING (PUT THIS HERE) ----------------
try:
    train_df = pd.read_csv("fraudTrain.csv")
    test_df = pd.read_csv("fraudTest.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["hour"] = df["trans_date_trans_time"].dt.hour

except Exception:
    st.warning("Dataset not available in Streamlit Cloud environment.")
    st.info("Please refer to the notebook and submission ZIP for full analysis.")
    st.stop()

# ---------------- ALL CHARTS GO BELOW ----------------

st.subheader("Fraud vs Legitimate Transactions (%)")
fraud_pct = df["is_fraud"].value_counts(normalize=True) * 100

fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraud"], fraud_pct.values)
ax.set_ylabel("Percentage (%)")
st.pyplot(fig)

# (other charts continue here)


# Sidebar filters
st.sidebar.header("Filters")
category = st.sidebar.multiselect(
    "Merchant Category",
    sorted(df["category"].unique())
)

if category:
    df = df[df["category"].isin(category)]

# ---- Chart 1
st.subheader("Fraud vs Legitimate Transactions (%)")
fraud_pct = df["is_fraud"].value_counts(normalize=True) * 100

fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraud"], fraud_pct.values)
ax.set_ylabel("Percentage (%)")
st.pyplot(fig)

# ---- Chart 2
st.subheader("Transaction Amount by Fraud Status")
fig, ax = plt.subplots()
sns.boxplot(x="is_fraud", y="amt", data=df, showfliers=False, ax=ax)
ax.set_yscale("log")
ax.set_xticklabels(["Legitimate", "Fraud"])
st.pyplot(fig)

# ---- Chart 3
st.subheader("Top 10 Merchant Categories by Fraud Rate")
cat_rate = (
    df.groupby("category")["is_fraud"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots()
cat_rate.plot(kind="bar", ax=ax)
st.pyplot(fig)

# ---- Chart 4
st.subheader("Fraud Rate by Hour of Day")
hour_rate = df.groupby("hour")["is_fraud"].mean()

fig, ax = plt.subplots()
hour_rate.plot(ax=ax)
st.pyplot(fig)

# ---- Chart 5
st.subheader("Fraud Rate by Age Group")
df["age_bin"] = pd.cut(df["age"], [18,25,35,45,55,65,100])
age_rate = df.groupby("age_bin")["is_fraud"].mean()

fig, ax = plt.subplots()
age_rate.plot(kind="bar", ax=ax)
st.pyplot(fig)

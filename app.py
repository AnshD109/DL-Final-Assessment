import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Analysis", layout="wide")
st.title("Credit Card Fraud Analysis Dashboard")

st.markdown("""
This dashboard explores patterns related to **fraudulent transactions** using EDA.
Fraud is **rare**, so after filtering it’s normal to sometimes see **0 fraud** in the selected subset.
When that happens, the dashboard will show you **counts** (not just rates) so it doesn’t look “broken”.
""")

# ---------- Data loading ----------
@st.cache_data
def load_df():
    # Try multiple filenames in case you renamed later
    for fname in ["fraud_streamlit.csv", "fraud_sample.csv", "fraudTrain.csv", "fraudTest.csv"]:
        try:
            df = pd.read_csv(fname)
            # If fraudTest.csv loads without required columns, it'll fail later and fall back to the next
            return df, fname
        except Exception:
            continue
    return None, None

df_raw, source_file = load_df()

if df_raw is None:
    st.error("No dataset file found in the repo. Expected one of: fraud_sample.csv / fraud_streamlit.csv / fraudTrain.csv.")
    st.stop()

required_cols = {"is_fraud", "amt", "category"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Dataset loaded from `{source_file}` but missing required columns: {sorted(list(missing))}")
    st.stop()

df = df_raw.copy()

# Ensure types
df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
df["amt"] = pd.to_numeric(df["amt"], errors="coerce")

# Optional datetime features if available
if "trans_date_trans_time" in df.columns:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["hour"] = df["trans_date_trans_time"].dt.hour
else:
    df["hour"] = np.nan

if "dob" in df.columns and "trans_date_trans_time" in df.columns:
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days // 365).astype("float")
else:
    df["age"] = np.nan

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

cat_options = sorted(df["category"].dropna().unique().tolist())
selected_cats = st.sidebar.multiselect("Merchant Category", cat_options, default=[])

# Amount filter
amt_min = float(np.nanmin(df["amt"])) if df["amt"].notna().any() else 0.0
amt_max = float(np.nanmax(df["amt"])) if df["amt"].notna().any() else 1.0
amt_range = st.sidebar.slider("Amount range", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max))

# Hour filter (only if available)
use_hour_filter = df["hour"].notna().any()
if use_hour_filter:
    hour_range = st.sidebar.slider("Hour range", min_value=0, max_value=23, value=(0, 23))
else:
    hour_range = (0, 23)

# Apply filters
df_f = df.copy()

if selected_cats:
    df_f = df_f[df_f["category"].isin(selected_cats)]

df_f = df_f[df_f["amt"].between(amt_range[0], amt_range[1], inclusive="both")]

if use_hour_filter:
    df_f = df_f[df_f["hour"].between(hour_range[0], hour_range[1], inclusive="both")]

if df_f.empty:
    st.error("Your filters returned **0 rows**. Remove some filters or widen the ranges.")
    st.stop()

# ---------- Metrics (this is what stops the '0 looks broken' problem) ----------
total_all = len(df)
fraud_all = int(df["is_fraud"].sum())
rate_all = (fraud_all / total_all) * 100 if total_all else 0

total_f = len(df_f)
fraud_f = int(df_f["is_fraud"].sum())
rate_f = (fraud_f / total_f) * 100 if total_f else 0

c1, c2, c3 = st.columns(3)
c1.metric("Rows (filtered)", f"{total_f:,}")
c2.metric("Fraud count (filtered)", f"{fraud_f:,}")
c3.metric("Fraud rate (filtered)", f"{rate_f:.4f}%")

st.caption(f"Data source in repo: `{source_file}` | Overall fraud in loaded data: {fraud_all:,} / {total_all:,} ({rate_all:.4f}%)")

if fraud_f == 0:
    st.warning(
        "In your **current filter selection**, there are **0 fraud transactions**. "
        "That’s why some fraud-rate plots look flat at 0. "
        "Try clearing category filters or widening the amount/hour range."
    )

st.markdown("""
### Source Code
- **GitHub Repo:** https://github.com/AnshD109/DL-Final-Assessment
""")

# ---------------- DASHBOARD LAYOUT (TABS) ----------------
tab1, tab2, tab3 = st.tabs(["Overview", "Patterns", "Segments"])

with tab1:
    st.subheader("Overview")
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", f"{len(df_f):,}")
    c2.metric("Fraud count", f"{int(df_f['is_fraud'].sum()):,}")
    rate = (df_f["is_fraud"].mean() * 100) if len(df_f) else 0
    c3.metric("Fraud rate", f"{rate:.4f}%")

    st.subheader("Fraud vs Legit (%)")
    # Chart 1
    fig, ax = plt.subplots()
    pct = df_f["is_fraud"].value_counts(normalize=True) * 100
    ax.bar(["Legitimate", "Fraud"], [pct.get(0, 0), pct.get(1, 0)])
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)

    st.subheader("Transaction Amount Distribution (Log)")
    # Chart 2
    fig, ax = plt.subplots()
    ax.hist(df_f["amt"].dropna(), bins=50)
    ax.set_yscale("log")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency (log)")
    st.pyplot(fig)

with tab2:
    st.subheader("Patterns")
    st.subheader("Amount vs Fraud (Log)")
    # Chart 3
    fig, ax = plt.subplots()
    sns.boxplot(x="is_fraud", y="amt", data=df_f, showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels(["Legitimate", "Fraud"])
    st.pyplot(fig)

    if "hour" in df_f.columns and df_f["hour"].notna().any():
        st.subheader("Fraud Rate by Hour")
        # Chart 4
        hourly = df_f.groupby("hour")["is_fraud"].mean() * 100
        fig, ax = plt.subplots()
        ax.plot(hourly.index, hourly.values, marker="o")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Fraud rate (%)")
        st.pyplot(fig)
    else:
        st.info("Hour feature not available in this dataset file.")

with tab3:
    st.subheader("Segments")

    st.subheader("Top Categories by Fraud Rate")
    # Chart 5
    grp = df_f.groupby("category")["is_fraud"].mean().sort_values(ascending=False).head(10) * 100
    fig, ax = plt.subplots()
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylabel("Fraud rate (%)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    if "age" in df_f.columns and df_f["age"].notna().sum() > 50:
        st.subheader("Fraud Rate by Age Group")
        # Chart 6
        tmp = df_f.copy()
        tmp["age_bin"] = pd.cut(tmp["age"], [18,25,35,45,55,65,100])
        age_rate = tmp.groupby("age_bin")["is_fraud"].mean() * 100
        fig, ax = plt.subplots()
        ax.bar(age_rate.index.astype(str), age_rate.values)
        ax.set_ylabel("Fraud rate (%)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("Age feature not available or insufficient valid values.")

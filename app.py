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

st.divider()

# ---------- Chart 1: Fraud vs Legit (%) ----------
st.subheader("Fraud vs Legitimate Transactions (%)")

pct = df_f["is_fraud"].value_counts(normalize=True) * 100
legit_pct = float(pct.get(0, 0.0))
fraud_pct = float(pct.get(1, 0.0))

fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraud"], [legit_pct, fraud_pct])
ax.set_ylabel("Percentage (%)")
ax.set_title("Class Imbalance (Filtered)")
for i, v in enumerate([legit_pct, fraud_pct]):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
st.pyplot(fig)

# ---------- Chart 2: Amount by fraud status (log boxplot) ----------
st.subheader("Transaction Amount by Fraud Status (Log Scale)")

fig, ax = plt.subplots()
# Only plot if both classes exist; otherwise seaborn still works but looks weird
sns.boxplot(x="is_fraud", y="amt", data=df_f, showfliers=False, ax=ax)
ax.set_yscale("log")
ax.set_xlabel("Class (0=Legit, 1=Fraud)")
ax.set_ylabel("Amount (log scale)")
ax.set_title("Amount Distribution by Class")
st.pyplot(fig)

# ---------- Chart 3: Top categories by fraud rate (with minimum count) ----------
st.subheader("Top Merchant Categories by Fraud Rate (stable only)")

min_tx = st.slider("Minimum transactions per category (to avoid misleading 0/100%)", 10, 300, 50, 10)

cat_stats = (
    df_f.groupby("category")["is_fraud"]
    .agg(tx_count="size", fraud_count="sum", fraud_rate="mean")
    .sort_values("fraud_rate", ascending=False)
)

cat_stats = cat_stats[cat_stats["tx_count"] >= min_tx].head(10)
cat_stats["fraud_rate_pct"] = cat_stats["fraud_rate"] * 100

if cat_stats.empty:
    st.info("No categories have enough transactions under your filters. Lower the minimum, or widen filters.")
else:
    fig, ax = plt.subplots()
    ax.bar(cat_stats.index.astype(str), cat_stats["fraud_rate_pct"].values)
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Category")
    ax.set_title(f"Top Categories by Fraud Rate (min {min_tx} tx)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    with st.expander("Show category table"):
        st.dataframe(cat_stats[["tx_count", "fraud_count", "fraud_rate_pct"]].sort_values("fraud_rate_pct", ascending=False))

# ---------- Chart 4: Fraud rate by hour + fraud counts (so flat 0 makes sense) ----------
st.subheader("Fraud by Hour of Day (Rate + Count)")

if not use_hour_filter:
    st.info("This dataset file has no `trans_date_trans_time`, so hour-based analysis is not available.")
else:
    hourly = df_f.groupby("hour")["is_fraud"].agg(tx_count="size", fraud_count="sum", fraud_rate="mean").sort_index()
    hourly["fraud_rate_pct"] = hourly["fraud_rate"] * 100

    fig, ax1 = plt.subplots()
    ax1.plot(hourly.index, hourly["fraud_rate_pct"].values, marker="o")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Fraud Rate (%)")
    ax1.set_title("Fraud Rate by Hour (Filtered)")

    ax2 = ax1.twinx()
    ax2.bar(hourly.index, hourly["fraud_count"].values, alpha=0.25)
    ax2.set_ylabel("Fraud Count")

    st.pyplot(fig)

    with st.expander("Show hourly table"):
        st.dataframe(hourly[["tx_count", "fraud_count", "fraud_rate_pct"]])

# ---------- Chart 5: Fraud by age group (rate + count) ----------
st.subheader("Fraud by Age Group (Rate + Count)")

if df_f["age"].notna().sum() < 50:
    st.info("Not enough valid age values in this dataset file to compute stable age-group analysis.")
else:
    bins = [18, 25, 35, 45, 55, 65, 100]
    df_f = df_f.copy()
    df_f["age_bin"] = pd.cut(df_f["age"], bins=bins)

    age_stats = df_f.groupby("age_bin")["is_fraud"].agg(tx_count="size", fraud_count="sum", fraud_rate="mean")
    age_stats["fraud_rate_pct"] = age_stats["fraud_rate"] * 100

    fig, ax = plt.subplots()
    ax.bar(age_stats.index.astype(str), age_stats["fraud_rate_pct"].values)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Age Group (Filtered)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.bar(age_stats.index.astype(str), age_stats["fraud_count"].values)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Fraud Count")
    ax.set_title("Fraud Count by Age Group (Filtered)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

st.markdown("---")
st.caption("If fraud-rate charts look flat at 0, it usually means your filtered subset contains 0 fraud rows. Check the Fraud Count metric above.")

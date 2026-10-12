"""
Dashboard — Step 7 of the project pipeline
===========================================
Minimal Streamlit dashboard showing, per drive: current risk score, the
protection action the decision layer chose, and why.

Run with:
    pip install streamlit pandas
    streamlit run dashboard.py

Right now it reads risk scores from a demo generator. To wire it to the
real pipeline, replace `load_risk_scores()` with either:
  (a) a DB query against your risk_scores table, or
  (b) a call to your teammate's saved model:
        bundle = joblib.load("baseline_risk_model.joblib")
        probs = bundle["model"].predict_proba(X)[:, 1]
"""

import random
from datetime import datetime

import pandas as pd
import streamlit as st

from db_utils import load_latest_risk_scores  # CHANGED: reads the real scores from PostgreSQL
from decision_layer import decide_batch, estimate_backup_load, POLICY_TIERS

st.set_page_config(page_title="Predictive Backup Scheduler", layout="wide")

TIER_COLORS = {
    "critical": "#c0392b",
    "high": "#e67e22",
    "moderate": "#f1c40f",
    "low": "#3498db",
    "nominal": "#27ae60",
}


# ----------------------------------------------------------------------
# Data loading  (REPLACE THIS with a real DB query or model call)
# ----------------------------------------------------------------------

@st.cache_data(ttl=60)  # CHANGED: new function. Tries the database first, falls back to the demo generator.
def load_risk_scores():
    """Real scores from the risk_scores table. If the database is unreachable or empty, use demo data
    and say so out loud -- a dashboard must never show invented numbers without telling you."""
    try:
        scores, factors = load_latest_risk_scores()
        if scores:
            return scores, factors, "PostgreSQL (risk_scores table)"
        reason = "risk_scores table is empty"
    except Exception as exc:  # connection refused, wrong password, tables missing, ...
        reason = f"database error: {type(exc).__name__}"
    scores, factors = load_demo_risk_scores()
    return scores, factors, f"DEMO DATA ({reason})"


@st.cache_data
def load_demo_risk_scores(n_drives: int = 40, seed: int = 7):  # CHANGED: renamed from load_risk_scores
    """
    Demo stand-in for real model output.

    Real version, querying the DB your teammate builds:

        import sqlalchemy
        engine = sqlalchemy.create_engine(DB_URL)
        df = pd.read_sql('''
            SELECT DISTINCT ON (serial_number)
                   serial_number, risk_score, computed_at
            FROM risk_scores
            ORDER BY serial_number, computed_at DESC
        ''', engine)
        return dict(zip(df.serial_number, df.risk_score)), {}
    """
    rng = random.Random(seed)
    scores, factors = {}, {}
    smart_names = [
        "smart_197_raw_mean (pending sectors)",
        "smart_5_raw_slope (reallocated sectors)",
        "smart_187_raw_mean (uncorrectable errors)",
        "smart_198_raw_slope (offline uncorrectable)",
        "disk_io_latency_ms_mean",
        "smart_188_raw_mean (command timeouts)",
    ]
    for i in range(n_drives):
        serial = f"DRIVE{i:04d}"
        # most drives healthy, a few degrading -- roughly matches real AFR
        score = rng.betavariate(1.2, 12) if rng.random() > 0.12 else rng.uniform(0.4, 0.97)
        scores[serial] = round(min(score, 0.99), 3)
        if scores[serial] >= 0.25:
            factors[serial] = rng.sample(smart_names, 3)
    return scores, factors


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("Predictive Failure Detection & Adaptive Backup Scheduling")
st.caption(
    "Risk scores come from the trained failure-prediction model; actions are "
    "assigned by the tiered decision policy. Refreshed "
    f"{datetime.now():%Y-%m-%d %H:%M}."
)

risk_scores, factors, data_source = load_risk_scores()  # CHANGED: also returns where the data came from
if data_source.startswith("DEMO"):
    st.warning(f"Showing {data_source}. These risk scores are randomly generated, not model output.")
else:
    st.caption(f"Data source: {data_source}")
decisions = decide_batch(risk_scores, factors=factors)
df = pd.DataFrame([d.to_row() for d in decisions])

# --- Sidebar filters -------------------------------------------------
st.sidebar.header("Filters")
tiers = ["critical", "high", "moderate", "low", "nominal"]
selected = st.sidebar.multiselect("Risk tier", tiers, default=tiers)
min_risk = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

st.sidebar.header("Current policy")
for threshold, tier, action, urgency, interval_h, _ in POLICY_TIERS:
    st.sidebar.markdown(
        f"**≥ {threshold:.2f}** — {tier}  \n"
        f"<span style='color:{TIER_COLORS[tier]}'>{action.value}</span>, every {interval_h}h",
        unsafe_allow_html=True,
    )

view = df[df.risk_tier.isin(selected) & (df.risk_score >= min_risk)]

# --- Summary metrics -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Drives monitored", len(df))
c2.metric("At risk (≥0.50)", int((df.risk_score >= 0.50).sum()))
c3.metric("Failover prep", int((df.risk_tier == "critical").sum()))
c4.metric("Mean risk", f"{df.risk_score.mean():.3f}")

# --- Fleet risk distribution ----------------------------------------
st.subheader("Risk distribution across the fleet")
tier_counts = (df.risk_tier.value_counts()
                 .reindex(tiers).fillna(0).astype(int))
st.bar_chart(tier_counts)

# --- Action queue ----------------------------------------------------
st.subheader("Action queue")
st.caption("Highest-risk drives first. These are the operations the scheduler will execute.")

if view.empty:
    st.info("No drives match the current filters.")
else:
    st.dataframe(
        view[["serial_number", "risk_score", "risk_tier", "action",
              "urgency", "backup_interval_hours", "next_backup_due"]],
        width="stretch",
        hide_index=True,
    )

# --- Per-drive explanation (the "auditable, not a black box" part) ----
st.subheader("Why this action?")
if not view.empty:
    pick = st.selectbox("Select a drive", view.serial_number.tolist())
    row = view[view.serial_number == pick].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<h2 style='color:{TIER_COLORS[row.risk_tier]};margin-bottom:0'>"
            f"{row.risk_score:.2f}</h2>"
            f"<p style='margin-top:0'>risk — <b>{row.risk_tier}</b></p>",
            unsafe_allow_html=True,
        )
        st.write(f"**Action:** {row.action}")
        st.write(f"**Urgency:** {row.urgency}")
        st.write(f"**Backup every:** {row.backup_interval_hours}h")
    with right:
        st.write("**Policy reason**")
        st.info(row.reason)
        st.write("**Top contributing signals**")
        if row.top_factors:
            for f in str(row.top_factors).split(", "):
                st.write(f"- {f}")
        else:
            st.caption("No dominant risk factors — drive is within normal range.")

# --- Cost justification ----------------------------------------------
st.subheader("Adaptive vs fixed-schedule backup load")
st.caption(
    "The core claim of the project: adaptive scheduling concentrates backup "
    "effort on drives that are actually degrading, instead of spreading it "
    "uniformly across a healthy fleet."
)
load = estimate_backup_load(decisions)
l1, l2, l3 = st.columns(3)
l1.metric("Adaptive ops / week", load["adaptive_backup_ops"])
l2.metric("Fixed daily ops / week", load["fixed_daily_backup_ops"],
          delta=f"{load['ops_change_pct']:+.1f}% vs adaptive", delta_color="off")
l3.metric("Extra coverage on at-risk drives",
          f"{load['high_risk_coverage_gain_pct']:+.0f}%")

with st.expander("Raw decision table (what gets written to the DB)"):
    st.dataframe(df, width="stretch", hide_index=True)

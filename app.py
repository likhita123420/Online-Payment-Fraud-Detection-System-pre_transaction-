import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Load Saved Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pipeline          = joblib.load('production_pipeline.pkl')
    fraud_model       = joblib.load('fraud_model.pkl')
    optimal_threshold = joblib.load('optimal_threshold.pkl')
    with open('feature_cols.json', 'r') as f:
        feature_cols  = json.load(f)
    return pipeline, fraud_model, optimal_threshold, feature_cols

pipeline, fraud_model, optimal_threshold, feature_cols = load_artifacts()

# ── Type Mapping ──────────────────────────────────────────────────────────────
type_map = {
    'CASH_OUT' : 0.8,
    'TRANSFER' : 0.8,
    'DEBIT'    : 0.3,
    'PAYMENT'  : 0.2,
    'CASH_IN'  : 0.1,
}

# ── Session History ───────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── App Title ─────────────────────────────────────────────────────────────────
st.title("💳 PaySim Fraud Detection System")
st.caption("Enter transaction details below to predict fraud probability.")
st.info(f"🎯 Optimal Decision Threshold: **{optimal_threshold:.4f}**")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step             = st.number_input("Step (Hour of Simulation)", min_value=1,   max_value=744, value=1)
    amount           = st.number_input("Transaction Amount (₹)",    min_value=0.0, value=1000.0,  step=100.0)
    old_balance_orig = st.number_input("Sender Opening Balance (₹)",min_value=0.0, value=5000.0,  step=100.0)

with col2:
    old_balance_dest = st.number_input("Receiver Opening Balance (₹)", min_value=0.0, value=2000.0, step=100.0)
    txn_type         = st.selectbox("Transaction Type", ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

st.divider()

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(step, amount, old_balance_orig, old_balance_dest, txn_type):

    # Time features
    step_hour  = step % 24
    step_day   = step // 24
    is_night   = 1 if step_hour < 6 or step_hour >= 22 else 0
    is_weekend = 1 if step_day % 7 in [5, 6] else 0

    # Amount features
    amount_log             = np.log1p(amount)
    balance_log            = np.log1p(old_balance_orig)
    amount_exceeds_balance = 1 if amount > old_balance_orig else 0

    # Ratio features
    amount_to_dest_balance = amount / (old_balance_dest + 1)
    type_ratio             = amount / (old_balance_orig + old_balance_dest + 1)

    # Zero balance flags
    orig_zero_before = 1 if old_balance_orig == 0 else 0
    dest_zero_before = 1 if old_balance_dest  == 0 else 0

    # Drain bin
    drain_ratio = amount / (old_balance_orig + 1)
    if drain_ratio == 0:
        drain_bin = 0
    elif drain_ratio <= 0.25:
        drain_bin = 1
    elif drain_ratio <= 0.50:
        drain_bin = 2
    elif drain_ratio <= 0.75:
        drain_bin = 3
    else:
        drain_bin = 4

    # Type encoding
    type_num = type_map.get(txn_type, 0)

    return {
        'oldbalanceDest'        : old_balance_dest,
        'step_hour'             : step_hour,
        'step_day'              : step_day,
        'is_night'              : is_night,
        'is_weekend'            : is_weekend,
        'drain_bin'             : drain_bin,
        'amount_to_dest_balance': amount_to_dest_balance,
        'orig_zero_before'      : orig_zero_before,
        'dest_zero_before'      : dest_zero_before,
        'amount_exceeds_balance': amount_exceeds_balance,
        'amount_log'            : amount_log,
        'balance_log'           : balance_log,
        'type_num'              : type_num,
        'type_ratio'            : type_ratio,
    }

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Fraud", use_container_width=True):

    features   = engineer_features(step, amount, old_balance_orig, old_balance_dest, txn_type)
    input_df   = pd.DataFrame([features])[feature_cols]

    fraud_prob = pipeline.predict_proba(input_df)[:, 1][0]
    is_fraud   = int(fraud_prob >= optimal_threshold)

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
    with col2:
        st.metric("Threshold Used",    f"{optimal_threshold:.4f}")
    with col3:
        st.metric("Decision", "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE")

    if is_fraud:
        st.error("🚨 **FRAUD DETECTED** — This transaction is flagged as fraudulent!")
    else:
        st.success("✅ **LEGITIMATE** — This transaction appears to be safe.")

    # Risk level
    if fraud_prob < 0.3:
        st.progress(fraud_prob, text="🟢 Low Risk")
    elif fraud_prob < 0.6:
        st.progress(fraud_prob, text="🟡 Medium Risk")
    else:
        st.progress(fraud_prob, text="🔴 High Risk")

    # Feature breakdown
    with st.expander("🔎 See Feature Breakdown"):
        feat_df = pd.DataFrame([features]).T.reset_index()
        feat_df.columns = ['Feature', 'Value']
        st.dataframe(feat_df, use_container_width=True)

    # Save to session history
    st.session_state.history.append({
        'Type'             : txn_type,
        'Amount (₹)'       : amount,
        'Fraud Probability': f"{fraud_prob*100:.2f}%",
        'Result'           : '🚨 Fraud' if is_fraud else '✅ Legitimate'
    })

# ── Transaction History ───────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Transaction History (This Session)")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    # Session summary
    total      = len(history_df)
    fraud_count = history_df['Result'].str.contains('Fraud').sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checked",  total)
    with col2:
        st.metric("Frauds Found",   fraud_count)
    with col3:
        st.metric("Fraud Rate",     f"{fraud_count/total*100:.1f}%")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.caption("No transactions checked yet in this session.")
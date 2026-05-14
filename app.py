"""
app.py — Streamlit fraud detection dashboard.

Prerequisites:
  1. Run `python train.py` first to generate ./artifacts/
  2. pip install streamlit tensorflow scikit-learn plotly pandas numpy

Run:
  streamlit run app.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense | Credit Card Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS (dark + blue accent matching your portfolio palette) ────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0a0f1e; color: #e2e8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #0d1528; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #111c35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
  }

  /* Section headers */
  h1, h2, h3 { color: #60a5fa; }

  /* Info / warning boxes */
  .stAlert { border-radius: 8px; }

  /* Tab active color */
  .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }

  /* Divider */
  hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ── LOAD ARTIFACTS ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    model   = keras.models.load_model("artifacts/model.keras")
    scaler  = pickle.load(open("artifacts/scaler.pkl", "rb"))
    meta    = json.load(open("artifacts/threshold.json"))
    X_test  = np.load("artifacts/X_test.npy")
    y_test  = np.load("artifacts/y_test.npy")
    return model, scaler, meta, X_test, y_test

try:
    model, scaler, meta, X_test, y_test = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_error = str(e)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudSense")
    st.markdown("Credit Card Fraud Detection  \nPowered by a Neural Network (MLP)")
    st.markdown("---")

    if artifacts_ok:
        st.success("Model loaded ✓")
        st.metric("Test PR-AUC", f"{meta['test_pr_auc']:.4f}")
        st.metric("Best threshold", f"{meta['best_threshold']:.4f}")
        st.metric("Target recall", f"{int(meta['target_recall']*100)}%")

        st.markdown("---")
        st.markdown("### Threshold tuner")
        threshold = st.slider(
            "Decision threshold",
            min_value=0.01, max_value=0.99,
            value=float(meta["best_threshold"]),
            step=0.01,
            help="Lower → catch more fraud (more false alarms). Higher → fewer false alarms (miss more fraud).",
        )
    else:
        st.error(f"Artifacts not found.\n\nRun `python train.py` first.\n\n{artifact_error}")
        threshold = 0.5

    st.markdown("---")
    st.caption("Dataset: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

# ── MAIN ────────────────────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("A production-style ML pipeline: Neural Network trained with class weighting + threshold optimization.")

if not artifacts_ok:
    st.error("Run `python train.py` to generate model artifacts, then reload this page.")
    st.stop()

# ── Pre-compute predictions (cached per threshold for speed) ──────────────────
@st.cache_data(show_spinner=False)
def get_probabilities(_model, _X_test):
    return _model.predict(_X_test, verbose=0).ravel()

y_prob = get_probabilities(model, X_test)
y_pred = (y_prob >= threshold).astype(int)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Evaluation", "🔍 Predictions", "⚙️ Threshold Analysis"])

# ─── TAB 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    n_test   = len(y_test)
    n_fraud  = int(y_test.sum())
    n_legit  = n_test - n_fraud

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test samples",     f"{n_test:,}")
    c2.metric("Legitimate",       f"{n_legit:,}")
    c3.metric("Fraud cases",      f"{n_fraud:,}")
    c4.metric("Fraud rate",       f"{n_fraud/n_test*100:.3f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Class Distribution (test set)")
        fig = px.pie(
            names=["Legitimate", "Fraud"],
            values=[n_legit, n_fraud],
            color_discrete_sequence=["#1e3a5f", "#ef4444"],
            hole=0.5,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Prediction Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_prob[y_test == 0], name="Legitimate",
            opacity=0.7, marker_color="#3b82f6", nbinsx=60
        ))
        fig.add_trace(go.Histogram(
            x=y_prob[y_test == 1], name="Fraud",
            opacity=0.8, marker_color="#ef4444", nbinsx=60
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"Threshold={threshold:.2f}", annotation_font_color="#fbbf24")
        fig.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="P(fraud)", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model Architecture")
    st.code("""
Input (30 features)  →  Dense(64, ReLU)  →  Dropout(0.3)
                     →  Dense(32, ReLU)  →  Dropout(0.2)
                     →  Dense(1, Sigmoid)
    """, language="text")


# ─── TAB 2: Evaluation ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Model Evaluation @ threshold = {threshold:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    df_report = pd.DataFrame(report).T
    df_report = df_report.drop(index=["accuracy"], errors="ignore")

    prec_fraud  = report.get("1", {}).get("precision", 0)
    rec_fraud   = report.get("1", {}).get("recall",    0)
    f1_fraud    = report.get("1", {}).get("f1-score",  0)
    roc_auc     = roc_auc_score(y_test, y_prob)
    pr_auc      = average_precision_score(y_test, y_prob)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision (Fraud)", f"{prec_fraud:.4f}")
    c2.metric("Recall (Fraud)",    f"{rec_fraud:.4f}")
    c3.metric("F1 (Fraud)",        f"{f1_fraud:.4f}")
    c4.metric("ROC-AUC",           f"{roc_auc:.4f}")
    c5.metric("PR-AUC",            f"{pr_auc:.4f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Confusion matrix
    with col_a:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Fraud", "Fraud"],
            y=["Not Fraud", "Fraud"],
            text_auto=True,
            color_continuous_scale=["#0a0f1e", "#1e3a5f", "#3b82f6"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # PR curve
    with col_b:
        st.markdown("#### Precision-Recall Curve")
        prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rec_curve, y=prec_curve, mode="lines",
            line=dict(color="#3b82f6", width=2), name=f"PR-AUC={pr_auc:.4f}"
        ))
        # Mark current operating point
        if threshold < max(thr_curve):
            idx = np.searchsorted(thr_curve[::-1], threshold)
            idx = len(thr_curve) - idx
            idx = min(idx, len(prec_curve) - 1)
            fig.add_trace(go.Scatter(
                x=[rec_curve[idx]], y=[prec_curve[idx]],
                mode="markers", marker=dict(size=12, color="#fbbf24", symbol="star"),
                name=f"Current thr={threshold:.2f}"
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    st.markdown("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             line=dict(color="#3b82f6", width=2), name=f"ROC-AUC={roc_auc:.4f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#4b5563", dash="dash"), name="Random"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis_title="FPR", yaxis_title="TPR",
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── TAB 3: Predictions ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Highest-Risk Transactions")
    st.markdown("Top flagged transactions from the test set, ranked by fraud probability.")

    top_n = st.slider("Show top N", min_value=5, max_value=100, value=20, step=5)

    results_df = pd.DataFrame({
        "Fraud Probability": y_prob,
        "True Label": y_test,
        "Predicted":  y_pred,
    })
    results_df["True Label"] = results_df["True Label"].map({0: "✅ Legit", 1: "🚨 Fraud"})
    results_df["Predicted"]  = results_df["Predicted"].map({0: "✅ Legit", 1: "🚨 Fraud"})
    results_df["Correct"]    = (results_df["True Label"] == results_df["Predicted"])
    results_df["Fraud Probability"] = results_df["Fraud Probability"].round(6)

    top_results = results_df.sort_values("Fraud Probability", ascending=False).head(top_n)
    st.dataframe(top_results, use_container_width=True)

    st.markdown("---")
    st.subheader("Predict a Custom Transaction")
    st.markdown("Enter scaled PCA feature values (V1–V28) + Time and Amount.")
    st.info("Note: Time and Amount will be scaled automatically using the trained scaler.")

    with st.expander("🔧 Manual Input", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_val   = st.number_input("Time (seconds)", value=0.0)
            amount_val = st.number_input("Amount ($)", value=100.0, min_value=0.0)
        with col2:
            v_vals = {}
            for i in range(1, 15):
                v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")
        v_vals2 = {}
        for i in range(15, 29):
            v_vals2[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

        if st.button("🔍 Predict", type="primary"):
            row = {"Time": time_val, "Amount": amount_val}
            row.update(v_vals)
            row.update(v_vals2)
            row_df = pd.DataFrame([row])
            row_df[["Time", "Amount"]] = scaler.transform(row_df[["Time", "Amount"]])
            X_input = row_df.values.astype("float32")
            prob = float(model.predict(X_input, verbose=0)[0][0])
            verdict = "🚨 FRAUD" if prob >= threshold else "✅ LEGITIMATE"
            color   = "#ef4444" if prob >= threshold else "#22c55e"
            st.markdown(f"""
            <div style='padding:20px; border-radius:12px; border: 2px solid {color}; text-align:center;'>
                <h2 style='color:{color};'>{verdict}</h2>
                <p style='font-size:1.2em;'>Fraud probability: <strong>{prob:.4f}</strong></p>
                <p>Threshold: {threshold:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("Test a Real Transaction")
    idx = st.number_input("Pick a transaction index (0 to {})".format(len(X_test)-1), 
                       min_value=0, max_value=len(X_test)-1, value=0)

    if st.button("Check This Transaction"):
        X_input = X_test[idx].reshape(1, -1)
        prob = float(model.predict(X_input, verbose=0)[0][0])
        true_label = y_test[idx]
        verdict = "🚨 FRAUD" if prob >= threshold else "✅ LEGITIMATE"
        actual  = "🚨 FRAUD" if true_label == 1 else "✅ LEGITIMATE"
        color   = "#ef4444" if prob >= threshold else "#22c55e"

        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; border: 2px solid {color}; text-align:center;'>
            <h2 style='color:{color};'>{verdict}</h2>
            <p>Fraud probability: <strong>{prob:.4f}</strong></p>
            <p>Actual label: <strong>{actual}</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 4: Threshold Analysis ────────────────────────────────────────────────
with tab4:
    st.subheader("Threshold Tradeoff Analysis")
    st.markdown("How precision and recall change across all possible thresholds.")

    prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_prob)
    thr_plot = np.append(thr_c, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thr_plot, y=prec_c, mode="lines",
                             name="Precision", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=thr_plot, y=rec_c, mode="lines",
                             name="Recall", line=dict(color="#ef4444", width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24",
                  annotation_text=f"Current={threshold:.2f}", annotation_font_color="#fbbf24")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="Threshold", yaxis_title="Score",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### At current threshold")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Threshold",  f"{threshold:.4f}")
    col2.metric("Precision",  f"{report.get('1',{}).get('precision',0):.4f}")
    col3.metric("Recall",     f"{report.get('1',{}).get('recall',0):.4f}")
    col4.metric("F1",         f"{report.get('1',{}).get('f1-score',0):.4f}")

    st.markdown("---")
    st.markdown("#### 💡 How to interpret this")
    st.markdown("""
- **High recall, low precision** → You catch almost all fraud, but flag many legitimate transactions (more false alarms). Good for high-stakes systems.
- **High precision, low recall** → Fewer false alarms, but you miss more actual fraud. Better for low-friction user experience.
- **Use the slider in the sidebar** to find the operating point that fits your business requirement.
    """)

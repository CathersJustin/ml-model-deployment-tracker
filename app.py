import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import drift
import metrics
import tracker

st.set_page_config(page_title="ML Model Deployment Tracker", layout="wide")


# Load data
registry = tracker.load_registry()
predictions = tracker.load_predictions()


st.title("ML Model Deployment Tracker")


# Sidebar: Add prediction
st.sidebar.header("Log Prediction")
if registry:
    model_options = [f"{m['name']} v{m['version']}" for m in registry]
    model_choice = st.sidebar.selectbox("Model", model_options)
    input_summary = st.sidebar.text_input("Input Summary")
    true_label = st.sidebar.text_input("True Label")
    prediction = st.sidebar.text_input("Prediction")
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    if st.sidebar.button("Log"):
        name, version = model_choice.split(' v')
        tracker.log_prediction(name, version, input_summary, prediction, true_label, confidence)
        st.sidebar.success("Logged prediction")
        st.experimental_rerun()
else:
    st.sidebar.info("No models in registry")


# Tabs
registry_tab, performance_tab, drift_tab, compare_tab, retrain_tab, log_tab = st.tabs([
    "Model Registry", "Performance Dashboard", "Drift Detection", "Model Comparison", "Retrain", "Prediction Log"])


with registry_tab:
    st.subheader("Registered Models")
    st.write(pd.DataFrame(registry))

with performance_tab:
    st.subheader("Model Performance")
    if not predictions.empty:
        metrics_list = []
        for (name, version), group in predictions.groupby(["Model Name", "Model Version"]):
            m = metrics.compute_classification_metrics(group["true_label"], group["prediction"])
            m.update({"model_name": name, "model_version": version})
            metrics_list.append(m)
        perf_df = pd.DataFrame(metrics_list)
        st.dataframe(perf_df)
        fig = px.bar(perf_df, x="model_version", y=["accuracy", "precision", "recall", "f1"], barmode="group", facet_col="model_name")
        st.plotly_chart(fig, use_container_width=True)
        # Alerts
        low_perf = perf_df[perf_df["accuracy"] < 0.85]
        if not low_perf.empty:
            st.warning("Models with low accuracy:\n" + low_perf.to_json(indent=2))
    else:
        st.info("No predictions logged yet.")

with drift_tab:
    st.subheader("Simple Drift Detection (PSI on confidence)")
    if not predictions.empty:
        # compare first half vs second half of confidence values
        half = len(predictions) // 2
        if half > 0:
            expected = predictions["confidence"].iloc[:half]
            actual = predictions["confidence"].iloc[half:]
            psi = drift.calculate_psi(expected, actual)
            st.write(f"PSI: {psi:.3f}")
            if drift.detect_drift(psi):
                st.error("Drift detected! Consider retraining.")
            else:
                st.success("No significant drift detected.")
        else:
            st.info("Not enough data for drift calculation")
    else:
        st.info("No predictions logged yet.")

with compare_tab:
    st.subheader("Compare Model Versions")
    if len(registry) >= 2:
        options = [f"{m['name']} v{m['version']}" for m in registry]
        col1, col2 = st.columns(2)
        with col1:
            choice_a = st.selectbox("Model A", options, key="a")
        with col2:
            choice_b = st.selectbox("Model B", options, key="b")
        if st.button("Compare"):
            name_a, ver_a = choice_a.split(' v')
            name_b, ver_b = choice_b.split(' v')
            pred_a = predictions[(predictions.model_name == name_a) & (predictions.model_version == ver_a)]
            pred_b = predictions[(predictions.model_name == name_b) & (predictions.model_version == ver_b)]
            if not pred_a.empty and not pred_b.empty:
                m_a = metrics.compute_classification_metrics(pred_a.true_label, pred_a.prediction)
                m_b = metrics.compute_classification_metrics(pred_b.true_label, pred_b.prediction)
                st.write("Model A", m_a)
                st.write("Model B", m_b)
            else:
                st.warning("Not enough data to compare")
    else:
        st.info("Need at least two model versions")

with retrain_tab:
    st.subheader("Simulate Retraining")
    name = st.text_input("Model Name")
    algorithm = st.text_input("Algorithm")
    if st.button("Retrain"):
        # For demo, just add a new version with random metrics
        metrics_dict = {"accuracy": round(pd.np.random.uniform(0.8, 0.99), 3)}
        new_ver = tracker.add_model_version(name, algorithm, metrics_dict)
        st.success(f"Added new version {new_ver} for {name}")
        st.experimental_rerun()

with log_tab:
    st.subheader("Prediction Log")
    st.write(predictions)
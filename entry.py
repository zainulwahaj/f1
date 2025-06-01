import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from pipeline import (
    load_master_data,
    preprocess_data,
    train_or_load_model,
    predict_position
)

# ─── STREAMLIT CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(page_title="F1 Qualifying Predictor", layout="wide")

def main():
    st.title("🏎️ F1 Qualifying Position Predictor")

    # ─── 1) Load Master Data ──────────────────────────────────────────────────────
    master_path = "./f1_master_data.csv"
    if not os.path.exists(master_path):
        st.error(f"Master data file '{master_path}' not found. Please place it here.")
        return

    master_df = load_master_data(master_path)

    # ─── 2) Preprocess to Features ────────────────────────────────────────────────
    features_df = preprocess_data(master_df)

    # ─── 3) Sidebar navigation ────────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select view:", ["🔎 EDA & Analytics", "🤖 Model & Predictions"])

    if page == "🔎 EDA & Analytics":
        st.header("🔍 Exploratory Data Analysis & Visuals")

        st.subheader("Dataset Preview")
        st.dataframe(features_df.head(10), use_container_width=True)
        st.write(f"**Dataset shape:** {features_df.shape}")

        st.subheader("Descriptive Statistics")
        stats = features_df.describe().transpose()
        st.dataframe(stats, use_container_width=True)

        st.subheader("Distribution of Qualifying Positions")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.histplot(features_df["position_qual"], bins=30, kde=True, ax=ax1)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        st.subheader("Feature Correlation Heatmap")
        numeric_df = features_df.select_dtypes(include=[np.number]).drop(columns=["position_qual"])
        corr = numeric_df.corr()
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax2)
        ax2.set_title("Correlation Matrix")
        st.pyplot(fig2)

        st.subheader("Top 5 Driver–Circuit Combos by Visits")
        top_combos = (
            features_df
            .groupby(["Driver.driverId", "Circuit.circuitId"])["Driver.driverId"]
            .count()
            .nlargest(5)
            .reset_index(name="visit_count")
        )
        st.dataframe(top_combos, use_container_width=True)

    else:  # 🤖 Model & Predictions
        st.header("🤖 Model Training & Prediction")

        # ─── 4) Train or load model & encoder ─────────────────────────────────────────
        te, model, mae, r2 = train_or_load_model(features_df)

        st.subheader("📈 Model Performance (In-Sample)")
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
        col2.metric("R² Score",              f"{r2:.3f}")

        st.markdown("---")
        st.subheader("🏷️ Predict Qualifying Position")
        driver_input      = st.text_input("Driver ID",      value="hamilton")
        circuit_input     = st.text_input("Circuit ID",     value="monza")
        constructor_input = st.text_input("Constructor ID", value="mercedes")

        if st.button("Predict"):
            try:
                pred = predict_position(
                    driver_input.strip(),
                    circuit_input.strip(),
                    constructor_input.strip(),
                    features_df,
                    te,
                    model
                )
                st.success(f"Predicted Qualifying Position → P{pred:.2f}")
            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()

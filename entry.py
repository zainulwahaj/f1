import os
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from eda import generate_eda_report
from pipeline import (
    load_master_data,
    preprocess_data,
    train_or_load_model,
    predict_position
)
import pickle
from pathlib import Path

# ─── STREAMLIT CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Qualifying Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF1801;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #CC1401;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoder():
    """Cache the model and encoder to avoid retraining"""
    try:
        model_path = Path("./models")
        model_path.mkdir(exist_ok=True)
        
        # Check if model exists
        if (model_path / "model.pkl").exists() and (model_path / "encoder.pkl").exists():
            with open(model_path / "model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(model_path / "encoder.pkl", "rb") as f:
                te = pickle.load(f)
            return te, model, None, None
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading cached model: {str(e)}")
        return None, None, None, None

@st.cache_data
def load_and_preprocess_data():
    """Cache the data loading and preprocessing"""
    try:
        master_path = "./f1_master_data.csv"
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Master data file '{master_path}' not found. Please place it here.")
        
        master_df = load_master_data(master_path)
        features_df = preprocess_data(master_df)
        return features_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    try:
        # ─── 1) Load Data and Model ───────────────────────────────────────────────
        features_df = load_and_preprocess_data()
        if features_df is None:
            return

        # Load or train model
        te, model, mae, r2 = load_model_and_encoder()
        if te is None or model is None:
            te, model, mae, r2 = train_or_load_model(features_df)

        # ─── 2) Sidebar Navigation ───────────────────────────────────────────────
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=200)
            st.title("Navigation")
            
            selected = option_menu(
                menu_title=None,
                options=["🏎️ Home", "📊 EDA Dashboard", "👨‍🏎️ Driver Analysis", "🏁 Circuit Analysis", "🔮 Predictions"],
                icons=["house", "graph-up", "person", "flag", "magic"],
                default_index=0,
            )

        # ─── 3) Home Page ───────────────────────────────────────────────────────
        if selected == "🏎️ Home":
            st.title("🏎️ F1 Qualifying Position Predictor")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                    ### Welcome to the F1 Qualifying Position Predictor!
                    
                    This application uses machine learning to predict Formula 1 qualifying positions
                    based on historical data and various features. Explore the data through our
                    comprehensive EDA dashboard or make predictions for upcoming races.
                    
                    #### Features:
                    - 📊 Interactive EDA Dashboard
                    - 🔮 Real-time Position Predictions
                    - 📈 Historical Performance Analysis
                    - 🏎️ Driver & Circuit Insights
                """)
            
            with col2:
                st.markdown("""
                    ### Quick Stats
                """)
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Total Races", len(features_df))
                    st.metric("Unique Drivers", features_df["Driver.driverId"].nunique())
                with col2_2:
                    st.metric("Unique Circuits", features_df["Circuit.circuitId"].nunique())
                    st.metric("Avg Position", f"{features_df['position_qual'].mean():.2f}")

        # ─── 4) EDA Dashboard ───────────────────────────────────────────────────
        elif selected == "📊 EDA Dashboard":
            try:
                st.title("📊 Exploratory Data Analysis")
                
                eda_report = generate_eda_report(features_df)
                
                # Driver Analysis
                st.header("👨‍🏎️ Driver Analysis")
                st.plotly_chart(eda_report['driver_analysis'], use_container_width=True)
                
                # Circuit Analysis
                st.header("🏁 Circuit Analysis")
                st.plotly_chart(eda_report['circuit_analysis'], use_container_width=True)
                
                # Feature Analysis
                st.header("📈 Feature Analysis")
                st.plotly_chart(eda_report['feature_analysis'], use_container_width=True)
            except Exception as e:
                st.error(f"Error generating EDA: {str(e)}")

        # ─── 5) Driver Analysis ────────────────────────────────────────────────
        elif selected == "👨‍🏎️ Driver Analysis":
            st.title("👨‍🏎️ Driver Performance Analysis")
            
            # Driver Selection
            selected_driver = st.selectbox(
                "Select Driver",
                options=sorted(features_df["Driver.driverId"].unique())
            )
            
            if selected_driver:
                driver_data = features_df[features_df["Driver.driverId"] == selected_driver]
                
                # Overview Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Races", len(driver_data))
                with col2:
                    st.metric("Average Position", f"{driver_data['position_qual'].mean():.2f}")
                with col3:
                    st.metric("Best Position", driver_data['position_qual'].min())
                with col4:
                    st.metric("Pole Positions", len(driver_data[driver_data['position_qual'] == 1]))
                
                # Performance Trends
                st.subheader("Performance Trends")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(driver_data))),
                    y=driver_data['position_qual'],
                    mode='lines+markers',
                    name='Qualifying Position'
                ))
                fig.update_layout(
                    title='Qualifying Position Trend',
                    xaxis_title='Races',
                    yaxis_title='Position',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Circuit Performance
                st.subheader("Circuit Performance")
                circuit_stats = driver_data.groupby('Circuit.circuitId').agg({
                    'position_qual': ['mean', 'min', 'count']
                }).round(2)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=circuit_stats.index,
                    y=circuit_stats[('position_qual', 'mean')],
                    text=circuit_stats[('position_qual', 'mean')].round(2),
                    textposition='auto',
                    name='Average Position'
                ))
                fig.update_layout(
                    title='Average Position by Circuit',
                    xaxis_title='Circuit',
                    yaxis_title='Average Position',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Constructor Performance
                st.subheader("Constructor Performance")
                constructor_stats = driver_data.groupby('constructorId').agg({
                    'position_qual': ['mean', 'min', 'count']
                }).round(2)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=constructor_stats.index,
                    y=constructor_stats[('position_qual', 'mean')],
                    text=constructor_stats[('position_qual', 'mean')].round(2),
                    textposition='auto',
                    name='Average Position'
                ))
                fig.update_layout(
                    title='Average Position by Constructor',
                    xaxis_title='Constructor',
                    yaxis_title='Average Position',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        # ─── 6) Circuit Analysis ───────────────────────────────────────────────
        elif selected == "🏁 Circuit Analysis":
            st.title("🏁 Circuit Performance Analysis")
            
            # Circuit Selection
            selected_circuit = st.selectbox(
                "Select Circuit",
                options=sorted(features_df["Circuit.circuitId"].unique())
            )
            
            if selected_circuit:
                circuit_data = features_df[features_df["Circuit.circuitId"] == selected_circuit]
                
                # Overview Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Races", len(circuit_data))
                with col2:
                    st.metric("Average Grid Size", f"{circuit_data['position_qual'].max().mean():.2f}")
                with col3:
                    st.metric("Most Poles", circuit_data[circuit_data['position_qual'] == 1]['Driver.driverId'].mode().iloc[0])
                with col4:
                    st.metric("DNF Rate", f"{(circuit_data['hist_n_dnfs'].mean() / circuit_data['hist_n_visits'].mean() * 100):.1f}%")
                
                # Position Distribution
                st.subheader("Position Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=circuit_data['position_qual'],
                    nbinsx=20,
                    name='Position Distribution'
                ))
                fig.update_layout(
                    title='Qualifying Position Distribution',
                    xaxis_title='Position',
                    yaxis_title='Frequency',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Driver Performance
                st.subheader("Driver Performance")
                driver_stats = circuit_data.groupby('Driver.driverId').agg({
                    'position_qual': ['mean', 'min', 'count']
                }).round(2)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=driver_stats.index,
                    y=driver_stats[('position_qual', 'mean')],
                    text=driver_stats[('position_qual', 'mean')].round(2),
                    textposition='auto',
                    name='Average Position'
                ))
                fig.update_layout(
                    title='Average Position by Driver',
                    xaxis_title='Driver',
                    yaxis_title='Average Position',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Constructor Performance
                st.subheader("Constructor Performance")
                constructor_stats = circuit_data.groupby('constructorId').agg({
                    'position_qual': ['mean', 'min', 'count']
                }).round(2)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=constructor_stats.index,
                    y=constructor_stats[('position_qual', 'mean')],
                    text=constructor_stats[('position_qual', 'mean')].round(2),
                    textposition='auto',
                    name='Average Position'
                ))
                fig.update_layout(
                    title='Average Position by Constructor',
                    xaxis_title='Constructor',
                    yaxis_title='Average Position',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        # ─── 7) Predictions Page ───────────────────────────────────────────────
        else:  # Predictions
            st.title("🔮 Qualifying Position Predictor")
            
            # Model Performance Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.3f}")
            with col2:
                st.metric("R² Score", f"{r2:.3f}")
            with col3:
                st.metric("Sample Size", len(features_df))
            
            st.markdown("---")
            
            # Prediction Interface
            st.subheader("Make a Prediction")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                driver_id = st.selectbox(
                    "Select Driver",
                    options=sorted(features_df["Driver.driverId"].unique())
                )
            
            with col2:
                circuit_id = st.selectbox(
                    "Select Circuit",
                    options=sorted(features_df["Circuit.circuitId"].unique())
                )
            
            with col3:
                constructor_id = st.selectbox(
                    "Select Constructor",
                    options=sorted(features_df["constructorId"].unique())
                )
            
            if st.button("🔮 Predict Qualifying Position"):
                try:
                    pred = predict_position(
                        driver_id,
                        circuit_id,
                        constructor_id,
                        features_df,
                        te,
                        model
                    )
                    
                    # Round prediction to whole number
                    pred_rounded = round(pred)
                    
                    # Historical Context
                    hist_data = features_df[
                        (features_df["Driver.driverId"] == driver_id) &
                        (features_df["Circuit.circuitId"] == circuit_id)
                    ]
                    
                    if not hist_data.empty:
                        st.markdown("---")
                        st.subheader("Historical Context")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Average Position",
                                f"{hist_data['position_qual'].mean():.2f}"
                            )
                        with col2:
                            st.metric(
                                "Best Position",
                                f"{hist_data['position_qual'].min():.0f}"
                            )
                        with col3:
                            st.metric(
                                "Previous Visits",
                                len(hist_data)
                            )
                        
                        # Historical Performance Plot
                        if len(hist_data) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(len(hist_data))),
                                y=hist_data['position_qual'],
                                mode='lines+markers',
                                name='Historical Positions'
                            ))
                            fig.update_layout(
                                title='Historical Performance at this Circuit',
                                xaxis_title='Previous Visits',
                                yaxis_title='Qualifying Position',
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No historical data available for this driver-circuit combination.")
                    
                    # Prediction Card
                    st.markdown("---")
                    st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background-color: #1E1E1E; border-radius: 10px;'>
                            <h2 style='color: white;'>Predicted Qualifying Position</h2>
                            <h1 style='color: #FF1801; font-size: 4rem;'>P{pred_rounded}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            
            # Batch Predictions
            st.markdown("---")
            st.subheader("Batch Predictions")
            
            uploaded_file = st.file_uploader(
                "Upload CSV with columns: Driver.driverId, Circuit.circuitId, constructorId",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    required_cols = ["Driver.driverId", "Circuit.circuitId", "constructorId"]
                    
                    if not all(col in batch_df.columns for col in required_cols):
                        st.error("CSV must contain columns: Driver.driverId, Circuit.circuitId, constructorId")
                        return
                    
                    if st.button("Run Batch Predictions"):
                        progress_bar = st.progress(0)
                        predictions = []
                        errors = []
                        
                        for i, row in batch_df.iterrows():
                            try:
                                pred = predict_position(
                                    row["Driver.driverId"],
                                    row["Circuit.circuitId"],
                                    row["constructorId"],
                                    features_df,
                                    te,
                                    model
                                )
                                predictions.append(round(pred))
                            except Exception as e:
                                predictions.append(None)
                                errors.append(f"Row {i+1}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(batch_df))
                        
                        batch_df["Predicted_Position"] = predictions
                        st.dataframe(batch_df)
                        
                        if errors:
                            st.warning("Some predictions failed:")
                            for error in errors:
                                st.error(error)
                        
                        # Download button
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "Download Predictions",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

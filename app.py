#!/usr/bin/env python3
"""
streamlit_app.py

Beautiful Streamlit UI for the F1 Qualifying Position Predictor.
Integrates data pipeline, EDA visualization, and prediction interface.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from pipeline import load_master_data, preprocess_data, train_or_load_model, predict_position

# Page configuration
st.set_page_config(
    page_title="🏎️ F1 Qualifying Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2196F3 0%, #21CBF3 100%);
    }
    
    .stAlert > div {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="F1 Qualifying Position Predictor",
        page_icon="🏎️",
        layout="wide"
    )
    
    st.title("🏎️ F1 Qualifying Position Predictor")
    
    try:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df = load_master_data()
            if df is None or df.empty:
                st.error("Failed to load data. Please check your data files.")
                return
                
            df = preprocess_data(df)
            if df is None or df.empty:
                st.error("Failed to preprocess data. Please check your data processing pipeline.")
                return
        
        # Train or load model
        with st.spinner("Loading prediction model..."):
            model = train_or_load_model(df)
            if model is None:
                st.error("Failed to load or train the model. Please check your model files.")
                return
        
        # Create sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["EDA Dashboard", "Prediction"])
        
        if page == "EDA Dashboard":
            show_eda_dashboard()
        else:
            show_prediction_page(df, model)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data files and model configuration.")

def show_home_page():
    st.markdown("## Welcome to the F1 Qualifying Position Predictor! 🏎️")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Pipeline
        - Fetch F1 data from Ergast API
        - Process and engineer features
        - Build master dataset
        """)
        
    with col2:
        st.markdown("""
        ### 📈 EDA Dashboard
        - Interactive visualizations
        - Data insights and patterns
        - Statistical analysis
        """)
        
    with col3:
        st.markdown("""
        ### 🎯 Predictions
        - Real-time qualifying predictions
        - Driver-circuit analysis
        - Performance insights
        """)
    
    # Quick status check
    st.markdown("---")
    st.markdown("### 🔍 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    files_to_check = [
        ("f1_master_data.csv", "Master Data"),
        ("f1_features.csv", "Features"),
        ("target_encoder.pkl", "Encoder"),
        ("qualifying_predictor_stack.pkl", "Model")
    ]
    
    for i, (file, name) in enumerate(files_to_check):
        with [col1, col2, col3, col4][i]:
            if os.path.exists(file):
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

def show_data_pipeline():
    st.markdown("## 📊 Data Pipeline Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Pipeline Steps:
        1. **Data Gathering**: Fetch F1 data from Ergast API
        2. **Preprocessing**: Engineer features and clean data
        3. **EDA**: Generate exploratory data analysis
        4. **Model Training**: Train stacked ensemble model
        """)
    
    with col2:
        seasons = st.multiselect(
            "Select seasons to process:",
            options=[2020, 2021, 2022, 2023, 2024],
            default=[2020, 2021, 2022, 2023, 2024]
        )
    
    # Pipeline execution
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        run_pipeline(seasons)
    
    st.markdown("---")
    
    # Individual components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Gather Data Only"):
            gather_data_only(seasons)
    
    with col2:
        if st.button("🔧 Process Features Only"):
            process_features_only()
    
    with col3:
        if st.button("🤖 Train Model Only"):
            train_model_only()

def run_pipeline(seasons):
    """Run the complete data pipeline with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Gathering
        status_text.text("📥 Gathering F1 data from Ergast API...")
        progress_bar.progress(25)
        
        if not os.path.exists("f1_master_data.csv"):
            master_df = load_master_data()
            master_df.to_csv("f1_master_data.csv", index=False)
        else:
            master_df = pd.read_csv("f1_master_data.csv")
        
        # Step 2: Feature Engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(50)
        
        features_df = preprocess_data(master_df)
        features_df.to_csv("f1_features.csv", index=False)
        
        # Step 3: Model Training
        status_text.text("🤖 Training machine learning model...")
        progress_bar.progress(75)
        
        train_or_load_model(features_df)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Pipeline completed successfully!")
        
        st.success(f"🎉 Pipeline completed! Processed {len(features_df)} records.")
        
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(features_df))
        with col2:
            st.metric("Unique Drivers", features_df['Driver.driverId'].nunique())
        with col3:
            st.metric("Unique Circuits", features_df['Circuit.circuitId'].nunique())
        with col4:
            st.metric("Seasons", len(seasons))
            
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        progress_bar.progress(0)

def gather_data_only(seasons):
    """Gather data only"""
    with st.spinner("Gathering F1 data..."):
        try:
            master_df = load_master_data()
            master_df.to_csv("f1_master_data.csv", index=False)
            st.success(f"✅ Master data saved! {len(master_df)} records")
        except Exception as e:
            st.error(f"❌ Data gathering failed: {str(e)}")

def process_features_only():
    """Process features only"""
    if not os.path.exists("f1_master_data.csv"):
        st.error("❌ Master data not found. Please gather data first.")
        return
    
    with st.spinner("Processing features..."):
        try:
            master_df = pd.read_csv("f1_master_data.csv")
            features_df = preprocess_data(master_df)
            features_df.to_csv("f1_features.csv", index=False)
            st.success(f"✅ Features processed! {len(features_df)} records")
        except Exception as e:
            st.error(f"❌ Feature processing failed: {str(e)}")

def train_model_only():
    """Train model only"""
    if not os.path.exists("f1_features.csv"):
        st.error("❌ Features not found. Please process features first.")
        return
    
    with st.spinner("Training model..."):
        try:
            features_df = pd.read_csv("f1_features.csv")
            train_or_load_model(features_df)
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")

def show_eda_dashboard():
    st.markdown("## 📈 Exploratory Data Analysis Dashboard")
    
    # Check if features exist
    if not os.path.exists("f1_features.csv"):
        st.warning("⚠️ Features dataset not found. Please run the data pipeline first.")
        return
    
    # Load data
    df = pd.read_csv("f1_features.csv")
    
    # Overview metrics
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Drivers", df['Driver.driverId'].nunique())
    with col3:
        st.metric("Unique Circuits", df['Circuit.circuitId'].nunique())
    with col4:
        st.metric("Avg Qualifying Position", f"{df['position_qual'].mean():.2f}")
    
    # Interactive visualizations
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Analysis", "👥 Driver Performance", "🏁 Circuit Analysis", "🔗 Feature Correlations"])
    
    with tab1:
        show_target_analysis(df)
    
    with tab2:
        show_driver_analysis(df)
    
    with tab3:
        show_circuit_analysis(df)
    
    with tab4:
        show_correlation_analysis(df)

def show_target_analysis(df):
    """Show target variable analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of qualifying positions
        fig = px.histogram(
            df, x='position_qual', 
            title='Distribution of Qualifying Positions',
            nbins=20,
            color_discrete_sequence=['#ff6b6b']
        )
        fig.update_layout(
            xaxis_title="Qualifying Position",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by position ranges
        df['position_range'] = pd.cut(df['position_qual'], 
                                    bins=[0, 3, 10, 20, float('inf')], 
                                    labels=['Top 3', '4-10', '11-20', '20+'])
        
        fig = px.box(
            df, x='position_range', y='position_qual',
            title='Position Distribution by Range',
            color='position_range',
            color_discrete_sequence=['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        )
        st.plotly_chart(fig, use_container_width=True)

def show_driver_analysis(df):
    st.subheader("Driver Performance Analysis")
    
    try:
        # Driver Performance Over Time
        driver_performance = df.groupby(['driver_name', 'year']).agg({
            'position': ['mean', 'count']
        }).round(2)
        driver_performance.columns = ['Average Position', 'Number of Races']
        driver_performance = driver_performance.reset_index()
        
        # Select top drivers for analysis
        top_drivers = df.groupby('driver_name')['position'].mean().nsmallest(10).index.tolist()
        driver_performance = driver_performance[driver_performance['driver_name'].isin(top_drivers)]
        
        # Create line plot for driver performance over time
        fig = px.line(
            driver_performance,
            x='year',
            y='Average Position',
            color='driver_name',
            title='Top 10 Drivers Performance Over Time',
            labels={
                'year': 'Year',
                'Average Position': 'Average Position',
                'driver_name': 'Driver'
            }
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Position",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Driver Consistency Analysis
        st.subheader("Driver Consistency Analysis")
        
        driver_consistency = df.groupby('driver_name').agg({
            'position': ['mean', 'std', 'count']
        }).round(2)
        driver_consistency.columns = ['Average Position', 'Position Std Dev', 'Number of Races']
        driver_consistency = driver_consistency.sort_values('Position Std Dev')
        
        # Visualize driver consistency
        fig = px.scatter(
            driver_consistency.reset_index(),
            x='Average Position',
            y='Position Std Dev',
            size='Number of Races',
            hover_name='driver_name',
            title='Driver Consistency Analysis',
            labels={
                'Average Position': 'Average Position',
                'Position Std Dev': 'Position Standard Deviation',
                'Number of Races': 'Number of Races'
            }
        )
        fig.update_layout(
            xaxis_title="Average Position",
            yaxis_title="Position Standard Deviation",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating driver analysis visualizations: {str(e)}")
        st.write("Please check if the data contains the required columns.")

def show_circuit_analysis(df):
    st.subheader("Circuit Analysis")
    
    # Circuit Performance Analysis
    circuit_stats = df.groupby('circuit_name').agg({
        'position': ['mean', 'std', 'count']
    }).round(2)
    circuit_stats.columns = ['Average Position', 'Position Std Dev', 'Number of Races']
    circuit_stats = circuit_stats.sort_values('Average Position')
    
    # Create a more readable display
    st.write("Circuit Performance Statistics")
    st.dataframe(circuit_stats.style.background_gradient(cmap='RdYlGn_r'))
    
    # Circuit Performance Visualization
    try:
        fig = px.bar(
            circuit_stats.reset_index(),
            x='circuit_name',
            y='Average Position',
            title='Average Position by Circuit',
            labels={'circuit_name': 'Circuit', 'Average Position': 'Average Position'},
            color='Average Position',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            xaxis_title="Circuit",
            yaxis_title="Average Position",
            xaxis={'tickangle': 45},
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Circuit Difficulty Analysis
        st.subheader("Circuit Difficulty Analysis")
        
        # Calculate circuit difficulty based on position variance
        circuit_difficulty = df.groupby('circuit_name').agg({
            'position': ['std', 'mean', 'count']
        }).round(2)
        circuit_difficulty.columns = ['Position Variance', 'Average Position', 'Number of Races']
        circuit_difficulty = circuit_difficulty.sort_values('Position Variance', ascending=False)
        
        # Visualize circuit difficulty
        fig = px.scatter(
            circuit_difficulty.reset_index(),
            x='Average Position',
            y='Position Variance',
            size='Number of Races',
            hover_name='circuit_name',
            title='Circuit Difficulty Analysis',
            labels={
                'Average Position': 'Average Position',
                'Position Variance': 'Position Variance',
                'Number of Races': 'Number of Races'
            }
        )
        fig.update_layout(
            xaxis_title="Average Position",
            yaxis_title="Position Variance",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating circuit analysis visualizations: {str(e)}")
        st.write("Please check if the data contains the required columns.")

def show_correlation_analysis(df):
    """Show feature correlation analysis"""
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with target
    target_corr = corr_matrix['position_qual'].abs().sort_values(ascending=False)[1:11]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Top Features Correlated with Qualifying Position")
        corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values
        })
        st.dataframe(corr_df, use_container_width=True)
    
    with col2:
        fig = px.bar(
            corr_df, x='Correlation', y='Feature',
            orientation='h',
            title='Feature Importance (Correlation)',
            color='Correlation',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_interface():
    st.markdown("## 🎯 Make Qualifying Position Predictions")
    
    # Check if model artifacts exist
    required_files = ["f1_features.csv", "target_encoder.pkl", "qualifying_predictor_stack.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("Please run the data pipeline first to generate the required files.")
        return
    
    # Load data for dropdown options
    df = pd.read_csv("f1_features.csv")
    
    st.markdown("### 🏎️ Select Driver, Circuit, and Constructor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drivers = sorted(df['Driver.driverId'].unique())
        selected_driver = st.selectbox("🏃 Select Driver:", drivers)
    
    with col2:
        circuits = sorted(df['Circuit.circuitId'].unique())
        selected_circuit = st.selectbox("🏁 Select Circuit:", circuits)
    
    with col3:
        constructors = sorted(df['constructorId'].unique())
        selected_constructor = st.selectbox("🏗️ Select Constructor:", constructors)
    
    # Prediction button
    if st.button("🔮 Predict Qualifying Position", type="primary"):
        try:
            # Load artifacts and make prediction
            features_df = pd.read_csv("f1_features.csv")
            te, model, mae, r2 = train_or_load_model(features_df)
            prediction = predict_position(
                selected_driver, selected_circuit, selected_constructor, 
                features_df, te, model
            )
            
            # Display prediction with style
            st.markdown(f"""
            <div class="prediction-card">
                🏁 Predicted Qualifying Position: P{prediction:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 📊 Historical Context")
            
            # Driver historical performance at this circuit
            driver_history = df[
                (df['Driver.driverId'] == selected_driver) & 
                (df['Circuit.circuitId'] == selected_circuit)
            ]
            
            if not driver_history.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_pos = driver_history['position_qual'].mean()
                    st.metric("Driver's Avg at Circuit", f"P{avg_pos:.2f}")
                
                with col2:
                    visits = len(driver_history)
                    st.metric("Previous Visits", visits)
                
                with col3:
                    best_pos = driver_history['position_qual'].min()
                    st.metric("Best Position", f"P{int(best_pos)}")
                
                # Performance trend
                if len(driver_history) > 1:
                    fig = px.line(
                        driver_history.sort_values(['season', 'round']),
                        y='position_qual',
                        title=f"{selected_driver} Performance at {selected_circuit}",
                        markers=True
                    )
                    fig.update_yaxis(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No historical data found for {selected_driver} at {selected_circuit}")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    
    # Batch predictions
    st.markdown("---")
    st.markdown("### 📋 Batch Predictions")
    
    if st.checkbox("Enable batch predictions"):
        uploaded_file = st.file_uploader(
            "Upload CSV with columns: Driver.driverId, Circuit.circuitId, constructorId",
            type=['csv']
        )
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            if st.button("🚀 Run Batch Predictions"):
                try:
                    features_df = pd.read_csv("f1_features.csv")
                    te, model, mae, r2 = train_or_load_model(features_df)
                    predictions = []
                    
                    progress_bar = st.progress(0)
                    for i, row in batch_df.iterrows():
                        pred = predict_position(
                            row['Driver.driverId'], 
                            row['Circuit.circuitId'], 
                            row['constructorId'],
                            features_df, te, model
                        )
                        predictions.append(pred)
                        progress_bar.progress((i + 1) / len(batch_df))
                    
                    batch_df['Predicted_Position'] = predictions
                    st.success("✅ Batch predictions completed!")
                    st.dataframe(batch_df)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_predictions.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Batch prediction failed: {str(e)}")

def show_about_page():
    st.markdown("## ℹ️ About F1 Qualifying Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🏎️ What is this app?
        
        The F1 Qualifying Position Predictor is a machine learning application that predicts where a driver will qualify for Formula 1 races based on historical performance data.
        
        ### 🔧 How it works:
        
        1. **Data Collection**: Fetches F1 data from the Ergast API
        2. **Feature Engineering**: Creates meaningful features from raw data including:
           - Driver historical performance at specific circuits
           - Rolling form indicators (last 3 and 5 races)
           - Constructor performance trends
           - Circuit difficulty metrics
           - Driver-circuit interaction features
        
        3. **Machine Learning**: Uses a stacked ensemble model combining:
           - CatBoost Regressor
           - LightGBM Regressor
           - Ridge Regression (meta-learner)
        
        4. **Target Encoding**: Handles categorical variables (drivers, circuits, constructors) using target encoding
        
        ### 📊 Key Features:
        - Interactive data pipeline management
        - Comprehensive EDA dashboard with Plotly visualizations
        - Real-time qualifying position predictions
        - Batch prediction capabilities
        - Historical performance analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Stack:
        - **Frontend**: Streamlit
        - **ML Libraries**: CatBoost, LightGBM, Scikit-learn
        - **Data Processing**: Pandas, NumPy
        - **Visualizations**: Plotly, Seaborn
        - **Data Source**: Ergast F1 API
        
        ### 📈 Model Performance:
        The model uses Mean Absolute Error (MAE) as the primary metric, typically achieving sub-position accuracy on qualifying predictions.
        
        ### 🚀 Getting Started:
        1. Run the data pipeline to gather F1 data
        2. Explore the EDA dashboard for insights
        3. Make predictions for driver-circuit combinations
        
        ### 🔄 Data Updates:
        The system can process data from 2020-2024 seasons. Run the pipeline periodically to incorporate new race data.
        """)

def show_constructor_analysis(df):
    st.subheader("Constructor Performance Analysis")
    
    try:
        # Constructor Performance Over Time
        constructor_performance = df.groupby(['constructor_name', 'year']).agg({
            'position': ['mean', 'count']
        }).round(2)
        constructor_performance.columns = ['Average Position', 'Number of Races']
        constructor_performance = constructor_performance.reset_index()
        
        # Select top constructors for analysis
        top_constructors = df.groupby('constructor_name')['position'].mean().nsmallest(5).index.tolist()
        constructor_performance = constructor_performance[constructor_performance['constructor_name'].isin(top_constructors)]
        
        # Create line plot for constructor performance over time
        fig = px.line(
            constructor_performance,
            x='year',
            y='Average Position',
            color='constructor_name',
            title='Top 5 Constructors Performance Over Time',
            labels={
                'year': 'Year',
                'Average Position': 'Average Position',
                'constructor_name': 'Constructor'
            }
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Position",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Constructor Consistency Analysis
        st.subheader("Constructor Consistency Analysis")
        
        constructor_consistency = df.groupby('constructor_name').agg({
            'position': ['mean', 'std', 'count']
        }).round(2)
        constructor_consistency.columns = ['Average Position', 'Position Std Dev', 'Number of Races']
        constructor_consistency = constructor_consistency.sort_values('Position Std Dev')
        
        # Visualize constructor consistency
        fig = px.scatter(
            constructor_consistency.reset_index(),
            x='Average Position',
            y='Position Std Dev',
            size='Number of Races',
            hover_name='constructor_name',
            title='Constructor Consistency Analysis',
            labels={
                'Average Position': 'Average Position',
                'Position Std Dev': 'Position Standard Deviation',
                'Number of Races': 'Number of Races'
            }
        )
        fig.update_layout(
            xaxis_title="Average Position",
            yaxis_title="Position Standard Deviation",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating constructor analysis visualizations: {str(e)}")
        st.write("Please check if the data contains the required columns.")

if __name__ == "__main__":
    main()
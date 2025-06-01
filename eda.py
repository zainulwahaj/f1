import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_target(df):
    """Analyze the target variable (qualifying position)"""
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Distribution of Qualifying Positions',
                                     'Box Plot by Position Ranges',
                                     'Cumulative Distribution',
                                     'Position vs Historical Average'))
    
    # Distribution
    fig.add_trace(
        go.Histogram(x=df['position_qual'], nbinsx=30, name='Position Distribution'),
        row=1, col=1
    )
    
    # Box plot by ranges
    df['position_range'] = pd.cut(df['position_qual'], 
                                bins=[0, 3, 10, 20, float('inf')],
                                labels=['Top 3', '4-10', '11-20', '20+'])
    fig.add_trace(
        go.Box(x=df['position_range'], y=df['position_qual'], name='Position Ranges'),
        row=1, col=2
    )
    
    # Cumulative distribution
    fig.add_trace(
        go.Scatter(x=np.sort(df['position_qual']),
                  y=np.arange(1, len(df) + 1) / len(df),
                  name='Cumulative Distribution'),
        row=2, col=1
    )
    
    # Position vs Historical Average
    fig.add_trace(
        go.Scatter(x=df['hist_avg_qpos'],
                  y=df['position_qual'],
                  mode='markers',
                  name='Position vs Historical Average'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def analyze_drivers(df):
    """Analyze driver performance patterns"""
    # Driver statistics
    driver_stats = (df.groupby('Driver.driverId')
                   .agg({
                       'position_qual': ['mean', 'std', 'min', 'max', 'count'],
                       'hist_avg_qpos': 'mean',
                       'hist_podium_rate': 'mean',
                       'hist_q3_rate': 'mean'
                   })
                   .round(2))
    
    # Sort by average position for top drivers
    top_drivers = driver_stats.sort_values(('position_qual', 'mean')).head(10)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top 10 Drivers by Average Position',
            'Driver Consistency Analysis',
            'Podium Rate',
            'Q3 Appearance Rate'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Driver Performance Overview
    fig.add_trace(
        go.Bar(
            x=top_drivers.index,
            y=top_drivers[('position_qual', 'mean')],
            name='Average Position',
            text=top_drivers[('position_qual', 'mean')].round(2),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Driver Consistency
    fig.add_trace(
        go.Scatter(
            x=driver_stats[('position_qual', 'mean')],
            y=driver_stats[('position_qual', 'std')],
            mode='markers',
            marker=dict(
                size=driver_stats[('position_qual', 'count')]/10,
                color=driver_stats[('position_qual', 'mean')],
                showscale=True,
                colorbar=dict(title='Avg Position')
            ),
            text=driver_stats.index,
            hoverinfo='text'
        ),
        row=1, col=2
    )
    
    # 3. Podium Rate
    fig.add_trace(
        go.Bar(
            x=top_drivers.index,
            y=top_drivers[('hist_podium_rate', 'mean')],
            name='Podium Rate',
            text=top_drivers[('hist_podium_rate', 'mean')].round(2),
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Q3 Rate
    fig.add_trace(
        go.Bar(
            x=top_drivers.index,
            y=top_drivers[('hist_q3_rate', 'mean')],
            name='Q3 Rate',
            text=top_drivers[('hist_q3_rate', 'mean')].round(2),
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text='Driver Performance Analysis',
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Driver', row=1, col=1)
    fig.update_yaxes(title_text='Average Position', row=1, col=1)
    fig.update_xaxes(title_text='Average Position', row=1, col=2)
    fig.update_yaxes(title_text='Position Standard Deviation', row=1, col=2)
    fig.update_xaxes(title_text='Driver', row=2, col=1)
    fig.update_yaxes(title_text='Podium Rate', row=2, col=1)
    fig.update_xaxes(title_text='Driver', row=2, col=2)
    fig.update_yaxes(title_text='Q3 Appearance Rate', row=2, col=2)
    
    return fig

def analyze_circuits(df):
    """Analyze circuit characteristics"""
    circuit_stats = (df.groupby('Circuit.circuitId')
                    .agg({
                        'position_qual': ['mean', 'std', 'min', 'max', 'count'],
                        'hist_avg_qpos': 'mean',
                        'hist_n_visits': 'mean',
                        'hist_n_dnfs': 'mean'
                    })
                    .round(2))
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Circuit Difficulty Analysis',
            'Average Position by Circuit',
            'Circuit Experience',
            'DNF Rate by Circuit'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Circuit Difficulty
    fig.add_trace(
        go.Scatter(
            x=circuit_stats[('position_qual', 'mean')],
            y=circuit_stats[('position_qual', 'std')],
            mode='markers',
            marker=dict(
                size=circuit_stats[('position_qual', 'count')]/10,
                color=circuit_stats[('position_qual', 'mean')],
                showscale=True,
                colorbar=dict(title='Avg Position')
            ),
            text=circuit_stats.index,
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # 2. Average Position
    fig.add_trace(
        go.Bar(
            x=circuit_stats.index,
            y=circuit_stats[('position_qual', 'mean')],
            name='Average Position',
            text=circuit_stats[('position_qual', 'mean')].round(2),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Circuit Experience
    fig.add_trace(
        go.Bar(
            x=circuit_stats.index,
            y=circuit_stats[('hist_n_visits', 'mean')],
            name='Experience',
            text=circuit_stats[('hist_n_visits', 'mean')].round(0),
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. DNF Rate
    dnf_rate = circuit_stats[('hist_n_dnfs', 'mean')] / circuit_stats[('hist_n_visits', 'mean')]
    fig.add_trace(
        go.Bar(
            x=circuit_stats.index,
            y=dnf_rate,
            name='DNF Rate',
            text=dnf_rate.round(2),
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text='Circuit Analysis',
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Average Position', row=1, col=1)
    fig.update_yaxes(title_text='Position Standard Deviation', row=1, col=1)
    fig.update_xaxes(title_text='Circuit', row=1, col=2)
    fig.update_yaxes(title_text='Average Position', row=1, col=2)
    fig.update_xaxes(title_text='Circuit', row=2, col=1)
    fig.update_yaxes(title_text='Number of Visits', row=2, col=1)
    fig.update_xaxes(title_text='Circuit', row=2, col=2)
    fig.update_yaxes(title_text='DNF Rate', row=2, col=2)
    
    return fig

def analyze_features(df):
    """Analyze feature correlations and importance"""
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Top correlated features with target
    target_corr = corr_matrix['position_qual'].abs().sort_values(ascending=False)
    top_features = target_corr[1:11]  # Exclude target itself
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Feature Correlation Heatmap',
            'Top 10 Features by Correlation',
            'Feature Distribution Analysis',
            'Feature Interaction Analysis'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0
        ),
        row=1, col=1
    )
    
    # 2. Top Features
    fig.add_trace(
        go.Bar(
            x=top_features.index,
            y=top_features.values,
            name='Feature Correlation',
            text=top_features.values.round(2),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Feature Distribution
    for feature in top_features.index[:5]:
        fig.add_trace(
            go.Box(
                y=df[feature],
                name=feature,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=2, col=1
        )
    
    # 4. Feature Interaction
    fig.add_trace(
        go.Scatter(
            x=df['hist_avg_qpos'],
            y=df['roll3_avg_qpos'],
            mode='markers',
            marker=dict(
                color=df['position_qual'],
                showscale=True,
                colorbar=dict(title='Position')
            ),
            name='Historical vs Recent Form'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text='Feature Analysis',
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Features', row=1, col=2)
    fig.update_yaxes(title_text='Correlation with Position', row=1, col=2)
    fig.update_xaxes(title_text='Features', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=1)
    fig.update_xaxes(title_text='Historical Average Position', row=2, col=2)
    fig.update_yaxes(title_text='Recent Form (3 races)', row=2, col=2)
    
    return fig

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    return {
        'target_analysis': analyze_target(df),
        'driver_analysis': analyze_drivers(df),
        'circuit_analysis': analyze_circuits(df),
        'feature_analysis': analyze_features(df)
    } 
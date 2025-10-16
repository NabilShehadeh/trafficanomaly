"""
Streamlit Dashboard for Network Traffic Anomaly Detection

This module provides:
1. Real-time anomaly monitoring dashboard
2. Network traffic statistics and visualizations
3. Alert management and history
4. Model performance metrics
5. System configuration interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import queue
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our custom modules
from data_collection.collector import NetworkDataCollector
from data_processing.processor import NetworkDataProcessor
from models.anomaly_detector import AnomalyDetector
from streaming.pipeline import RealTimePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardManager:
    """Main dashboard manager class"""
    
    def __init__(self):
        self.data_collector = NetworkDataCollector()
        self.processor = NetworkDataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.pipeline = None
        self.alerts_history = []
        self.traffic_stats = []
        self.model_metrics = {}
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'traffic_data' not in st.session_state:
            st.session_state.traffic_data = pd.DataFrame()
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic network traffic data
        data = {
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                     periods=n_samples, freq='1min'),
            'src_ip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" 
                      for _ in range(n_samples)],
            'dst_ip': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" 
                      for _ in range(n_samples)],
            'src_port': np.random.randint(1, 65535, n_samples),
            'dst_port': np.random.randint(1, 65535, n_samples),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
            'packet_length': np.random.randint(64, 1500, n_samples),
            'duration': np.random.exponential(0.1, n_samples),
            'flow_bytes_sent': np.random.randint(100, 10000, n_samples),
            'flow_bytes_received': np.random.randint(100, 10000, n_samples),
            'packets_sent': np.random.randint(1, 100, n_samples),
            'packets_received': np.random.randint(1, 100, n_samples),
            'is_anomaly': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        return pd.DataFrame(data)
    
    def create_traffic_overview(self, df: pd.DataFrame):
        """Create traffic overview section"""
        st.subheader("📊 Network Traffic Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_packets = len(df)
            st.metric("Total Packets", f"{total_packets:,}")
        
        with col2:
            anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            st.metric("Anomalies Detected", f"{anomalies:,}")
        
        with col3:
            unique_ips = df['src_ip'].nunique() if 'src_ip' in df.columns else 0
            st.metric("Unique Source IPs", f"{unique_ips:,}")
        
        with col4:
            avg_packet_size = df['packet_length'].mean() if 'packet_length' in df.columns else 0
            st.metric("Avg Packet Size", f"{avg_packet_size:.0f} bytes")
    
    def create_traffic_visualizations(self, df: pd.DataFrame):
        """Create traffic visualizations"""
        st.subheader("📈 Traffic Visualizations")
        
        if df.empty:
            st.warning("No data available for visualization")
            return
        
        # Time series of packet counts
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df.groupby('hour').size().reset_index(name='packet_count')
            
            fig1 = px.line(hourly_counts, x='hour', y='packet_count', 
                          title="Packets per Hour", markers=True)
            st.plotly_chart(fig1, use_container_width=True)
        
        # Protocol distribution
        if 'protocol' in df.columns:
            protocol_counts = df['protocol'].value_counts()
            
            fig2 = px.pie(values=protocol_counts.values, names=protocol_counts.index,
                         title="Protocol Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Packet size distribution
        if 'packet_length' in df.columns:
            fig3 = px.histogram(df, x='packet_length', nbins=50,
                               title="Packet Size Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Top source IPs
        if 'src_ip' in df.columns:
            top_ips = df['src_ip'].value_counts().head(10)
            
            fig4 = px.bar(x=top_ips.values, y=top_ips.index,
                         orientation='h', title="Top 10 Source IPs")
            fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)
    
    def create_anomaly_analysis(self, df: pd.DataFrame):
        """Create anomaly analysis section"""
        st.subheader("🚨 Anomaly Analysis")
        
        if 'is_anomaly' not in df.columns:
            st.warning("No anomaly data available")
            return
        
        anomalies_df = df[df['is_anomaly'] == 1]
        
        if anomalies_df.empty:
            st.success("✅ No anomalies detected!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Anomaly Rate", f"{len(anomalies_df)/len(df)*100:.2f}%")
        
        with col2:
            st.metric("Total Anomalies", len(anomalies_df))
        
        # Anomaly timeline
        if 'timestamp' in anomalies_df.columns:
            anomalies_df['hour'] = anomalies_df['timestamp'].dt.hour
            hourly_anomalies = anomalies_df.groupby('hour').size().reset_index(name='anomaly_count')
            
            fig = px.bar(hourly_anomalies, x='hour', y='anomaly_count',
                        title="Anomalies by Hour")
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details table
        st.subheader("Anomaly Details")
        
        # Select relevant columns for display
        display_columns = ['timestamp', 'src_ip', 'dst_ip', 'protocol', 'packet_length', 'duration']
        available_columns = [col for col in display_columns if col in anomalies_df.columns]
        
        if available_columns:
            st.dataframe(anomalies_df[available_columns].head(20), use_container_width=True)
    
    def create_model_performance(self):
        """Create model performance section"""
        st.subheader("🤖 Model Performance")
        
        if not st.session_state.model_trained:
            st.warning("No model trained yet. Train a model first.")
            return
        
        # Generate sample metrics (in real implementation, these would come from actual model evaluation)
        metrics = {
            'Isolation Forest': {'accuracy': 0.92, 'precision': 0.88, 'recall': 0.85, 'f1': 0.86},
            'Autoencoder': {'accuracy': 0.89, 'precision': 0.82, 'recall': 0.90, 'f1': 0.86},
            'K-Means': {'accuracy': 0.87, 'precision': 0.80, 'recall': 0.88, 'f1': 0.84},
            'Ensemble': {'accuracy': 0.94, 'precision': 0.91, 'recall': 0.89, 'f1': 0.90}
        }
        
        # Create metrics comparison chart
        models = list(metrics.keys())
        accuracy_scores = [metrics[model]['accuracy'] for model in models]
        f1_scores = [metrics[model]['f1'] for model in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=accuracy_scores, name="Accuracy", marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name="F1-Score", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df, use_container_width=True)
    
    def create_alerts_section(self):
        """Create alerts section"""
        st.subheader("🚨 Live Alerts")
        
        # Simulate real-time alerts
        if st.button("Generate Sample Alert"):
            alert = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'severity': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
                'type': 'Network Anomaly',
                'src_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'dst_ip': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'description': 'Unusual traffic pattern detected',
                'status': 'ACTIVE'
            }
            st.session_state.alerts.append(alert)
        
        # Display alerts
        if st.session_state.alerts:
            alerts_df = pd.DataFrame(st.session_state.alerts)
            
            # Filter by severity
            severity_filter = st.selectbox("Filter by Severity", 
                                         ['ALL'] + list(alerts_df['severity'].unique()))
            
            if severity_filter != 'ALL':
                alerts_df = alerts_df[alerts_df['severity'] == severity_filter]
            
            # Display alerts table
            st.dataframe(alerts_df, use_container_width=True)
            
            # Alert statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_alerts = len(alerts_df)
                st.metric("Total Alerts", total_alerts)
            
            with col2:
                high_severity = len(alerts_df[alerts_df['severity'].isin(['HIGH', 'CRITICAL'])])
                st.metric("High Severity", high_severity)
            
            with col3:
                active_alerts = len(alerts_df[alerts_df['status'] == 'ACTIVE'])
                st.metric("Active Alerts", active_alerts)
        else:
            st.info("No alerts available")
    
    def create_system_config(self):
        """Create system configuration section"""
        st.subheader("⚙️ System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Collection Settings**")
            use_synthetic = st.checkbox("Use Synthetic Data", value=True)
            packet_count = st.slider("Packets per Capture", 10, 1000, 100)
            capture_interval = st.slider("Capture Interval (seconds)", 1, 60, 5)
        
        with col2:
            st.write("**Model Settings**")
            contamination = st.slider("Anomaly Contamination", 0.01, 0.2, 0.1)
            encoding_dim = st.slider("Autoencoder Encoding Dim", 8, 64, 32)
            n_clusters = st.slider("K-Means Clusters", 2, 20, 8)
        
        # Pipeline controls
        st.write("**Pipeline Controls**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Pipeline"):
                st.session_state.pipeline_running = True
                st.success("Pipeline started!")
        
        with col2:
            if st.button("⏹️ Stop Pipeline"):
                st.session_state.pipeline_running = False
                st.warning("Pipeline stopped!")
        
        with col3:
            if st.button("🔄 Restart Pipeline"):
                st.session_state.pipeline_running = False
                time.sleep(1)
                st.session_state.pipeline_running = True
                st.info("Pipeline restarted!")
        
        # Status indicator
        if st.session_state.pipeline_running:
            st.success("🟢 Pipeline Status: RUNNING")
        else:
            st.error("🔴 Pipeline Status: STOPPED")

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Network Traffic Anomaly Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard manager
    dashboard = DashboardManager()
    dashboard.initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🛡️ Network Security Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["Dashboard", "Traffic Analysis", "Anomaly Detection", "Model Performance", "Alerts", "Configuration"]
    )
    
    # Generate sample data if not exists
    if st.session_state.traffic_data.empty:
        st.session_state.traffic_data = dashboard.generate_sample_data()
    
    # Main content based on selected page
    if page == "Dashboard":
        st.title("🛡️ Network Traffic Anomaly Detection Dashboard")
        st.markdown("Real-time monitoring and analysis of network traffic anomalies")
        
        # Overview metrics
        dashboard.create_traffic_overview(st.session_state.traffic_data)
        
        # Quick visualizations
        dashboard.create_traffic_visualizations(st.session_state.traffic_data)
        
        # Recent alerts
        st.subheader("🚨 Recent Alerts")
        if st.session_state.alerts:
            recent_alerts = st.session_state.alerts[-5:]  # Last 5 alerts
            for alert in recent_alerts:
                severity_color = {
                    'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'
                }
                st.write(f"{severity_color.get(alert['severity'], '⚪')} **{alert['severity']}**: {alert['description']}")
        else:
            st.info("No recent alerts")
    
    elif page == "Traffic Analysis":
        st.title("📊 Traffic Analysis")
        
        dashboard.create_traffic_overview(st.session_state.traffic_data)
        dashboard.create_traffic_visualizations(st.session_state.traffic_data)
    
    elif page == "Anomaly Detection":
        st.title("🚨 Anomaly Detection")
        
        # Train model button
        if st.button("🤖 Train Anomaly Detection Models"):
            with st.spinner("Training models..."):
                # Generate sample data for training
                train_data = dashboard.generate_sample_data(5000)
                processed_data = dashboard.processor.parse_cicids2017_format(train_data)
                X, y = dashboard.processor.prepare_features_for_ml(processed_data)
                
                if not X.empty:
                    # Train ensemble model
                    results = dashboard.anomaly_detector.train_ensemble_model(X, y)
                    st.session_state.model_trained = True
                    st.success("Models trained successfully!")
                else:
                    st.error("Failed to prepare data for training")
        
        dashboard.create_anomaly_analysis(st.session_state.traffic_data)
    
    elif page == "Model Performance":
        st.title("🤖 Model Performance")
        dashboard.create_model_performance()
    
    elif page == "Alerts":
        st.title("🚨 Alert Management")
        dashboard.create_alerts_section()
    
    elif page == "Configuration":
        st.title("⚙️ System Configuration")
        dashboard.create_system_config()
    
    # Footer
    st.markdown("---")
    st.markdown("**Network Traffic Anomaly Detection System** | Built with Streamlit, Kafka, Spark, and ML")

if __name__ == "__main__":
    main()

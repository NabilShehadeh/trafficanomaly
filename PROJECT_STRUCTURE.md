# Project Structure

This document describes the structure of the Network Traffic Anomaly Detection System.

```
traffic-anomaly-detection/
├── .github/                    # GitHub configuration
│   ├── workflows/              # GitHub Actions workflows
│   └── ISSUE_TEMPLATE/         # Issue templates
├── config/                     # Configuration files
│   └── settings.py             # Main configuration
├── data/                       # Data storage
│   ├── cicids2017/             # CICIDS2017 dataset
│   └── captured/               # Live packet captures
├── models/                     # Trained models
├── logs/                       # Log files
├── src/                        # Source code
│   ├── data_collection/        # Data collection modules
│   │   └── collector.py        # Main data collector
│   ├── data_processing/        # Data processing modules
│   │   └── processor.py        # Data processor
│   ├── models/                 # ML models
│   │   ├── anomaly_detector.py # Anomaly detection models
│   │   └── train_models.py     # Model training script
│   ├── streaming/              # Real-time processing
│   │   └── pipeline.py         # Streaming pipeline
│   └── dashboard/              # Dashboard
│       └── app.py              # Streamlit dashboard
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── setup.py                    # Setup script
├── trafficanomaly.py          # Main entry point
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # Contributing guidelines
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

## Key Components

### Data Collection (`src/data_collection/`)
- **collector.py**: Main data collection module
  - CICIDS2017 dataset integration
  - Live packet capture via Scapy
  - Synthetic data generation

### Data Processing (`src/data_processing/`)
- **processor.py**: Data processing and feature engineering
  - CICIDS2017 format parsing
  - Packet capture processing
  - Feature extraction and engineering
  - Data validation and cleaning

### Models (`src/models/`)
- **anomaly_detector.py**: Anomaly detection algorithms
  - Isolation Forest
  - Autoencoder (Deep Learning)
  - K-Means and DBSCAN clustering
  - Ensemble methods
- **train_models.py**: Model training script

### Streaming (`src/streaming/`)
- **pipeline.py**: Real-time processing pipeline
  - Kafka producer for data streaming
  - Spark streaming consumer
  - Real-time anomaly detection
  - Alert generation

### Dashboard (`src/dashboard/`)
- **app.py**: Streamlit dashboard
  - Real-time monitoring
  - Traffic visualizations
  - Alert management
  - Model performance metrics

### Configuration (`config/`)
- **settings.py**: Centralized configuration
  - Kafka and Spark settings
  - Model parameters
  - Environment-specific configs

## Data Flow

1. **Data Collection**: Network traffic data is collected from various sources
2. **Data Processing**: Raw data is parsed and features are engineered
3. **Model Training**: Anomaly detection models are trained on processed data
4. **Real-time Processing**: Live data flows through Kafka and Spark
5. **Anomaly Detection**: Trained models detect anomalies in real-time
6. **Alert Generation**: Alerts are generated for detected anomalies
7. **Dashboard Display**: Results are visualized in the Streamlit dashboard

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning (for autoencoders)

### Streaming Dependencies
- **kafka-python**: Kafka client
- **pyspark**: Apache Spark integration

### Network Analysis
- **scapy**: Packet manipulation and capture
- **pyshark**: Wireshark integration

### Dashboard
- **streamlit**: Web dashboard framework
- **plotly**: Interactive visualizations

### Optional Dependencies
- **pyod**: Additional anomaly detection algorithms
- **dpkt**: Packet parsing utilities

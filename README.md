# Real-Time Network Traffic Anomaly Detection System

A comprehensive system for detecting network anomalies in real-time using machine learning and streaming data processing.

## Features

- **Real-time Data Processing**: Kafka + Spark/Flink pipeline for streaming network traffic analysis
- **Multiple Data Sources**: Support for CICIDS2017 dataset and live packet capture via Wireshark
- **Advanced Anomaly Detection**: Isolation Forest, Autoencoder, and clustering-based models
- **Live Dashboard**: Streamlit-based dashboard with real-time alerts and statistics
- **Network Protocol Analysis**: Deep packet inspection and OSI model compliance

## Project Structure

```
traffic-anomaly-detection/
├── data/
│   ├── cicids2017/          # CICIDS2017 dataset
│   └── captured/            # Live packet captures
├── src/
│   ├── data_collection/     # Data gathering modules
│   ├── data_processing/     # ETL and parsing
│   ├── models/             # ML models for anomaly detection
│   ├── streaming/          # Real-time processing pipeline
│   └── dashboard/          # Streamlit dashboard
├── config/
│   └── settings.py         # Configuration settings
├── tests/                  # Unit tests
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd traffic-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download CICIDS2017 dataset (optional):
```bash
python src/data_collection/download_cicids2017.py
```

## Usage

### Start the Real-time Pipeline
```bash
python src/streaming/kafka_producer.py
python src/streaming/spark_processor.py
```

### Launch the Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Train Models
```bash
python src/models/train_models.py
```

## Skills Demonstrated

- **In-stream Data Processing**: Real-time analysis using Kafka and Spark
- **Predictive Modeling**: Multiple ML algorithms for anomaly detection
- **Anomaly Detection**: Advanced techniques for identifying network intrusions
- **Networking Principles**: OSI model compliance and packet analysis
- **ETL Pipeline**: Extract, Transform, Load operations for network data
- **Streaming Analytics**: Live monitoring and alerting capabilities

## Technologies Used

- **Data Processing**: Apache Spark, Apache Flink, Kafka
- **Machine Learning**: scikit-learn, Isolation Forest, Autoencoders
- **Network Analysis**: Scapy, PyShark, dpkt
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Data Sources**: CICIDS2017, Wireshark packet captures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

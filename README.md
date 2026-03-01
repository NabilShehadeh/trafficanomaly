# Real-Time Network Traffic Anomaly Detection System

A comprehensive system for detecting network anomalies in real-time using machine learning and streaming data processing.

---

## Interactive dashboard (live on first click)

**Try the project immediately — no install required.** Open the dashboard to explore an interactive graph, histogram, and heatmap built from the same pipeline (traffic volume over time, packet and anomaly score distributions, feature correlations).

[![Open Interactive Dashboard](https://img.shields.io/badge/Open-Interactive%20Dashboard-2088FF?style=for-the-badge&logo=github)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/NabilShehadeh/trafficanomaly/main/docs/index.html)

**[Open interactive dashboard](https://htmlpreview.github.io/?https://raw.githubusercontent.com/NabilShehadeh/trafficanomaly/main/docs/index.html)**

Graph, histogram and heatmap in the browser. You can also open `docs/index.html` locally after cloning the repo.

**Quick local run (after clone):**
```bash
pip install -r requirements.txt   # or requirements-ci.txt for minimal deps
python trafficanomaly.py test      # run tests
python trafficanomaly.py collect   # generate sample data
python trafficanomaly.py train     # train models
streamlit run src/dashboard/app.py # launch interactive dashboard
```

---

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

All-in-one CLI (recommended):

| Command | Description |
|--------|--------------|
| `python trafficanomaly.py test` | Run built-in tests |
| `python trafficanomaly.py collect` | Generate synthetic traffic data |
| `python trafficanomaly.py train` | Train anomaly detection models |
| `python trafficanomaly.py dashboard` | Launch Streamlit dashboard |
| `python trafficanomaly.py pipeline` | Start real-time pipeline (requires Kafka) |

### Launch the Dashboard (interactive UI)
```bash
streamlit run src/dashboard/app.py
```
Or: `python trafficanomaly.py dashboard`

### Train Models
```bash
python trafficanomaly.py train
# or: python src/models/train_models.py
```

### Real-time Pipeline (optional; requires Kafka + Spark)
```bash
python trafficanomaly.py pipeline
```

## Technologies Used

- **Data Processing**: Apache Spark, Apache Flink, Kafka
- **Machine Learning**: scikit-learn, Isolation Forest, Autoencoders
- **Network Analysis**: Scapy, PyShark, dpkt
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Data Sources**: CICIDS2017, Wireshark packet captures
  
## License

MIT License - see LICENSE file for details

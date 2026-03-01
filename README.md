# Real-Time Network Traffic Anomaly Detection System

A comprehensive system for detecting network anomalies in real-time using machine learning and streaming data processing.

---

## Try it now

### Interactive dashboard (graph, histogram, heatmap)

When the repo is published on GitHub you get:

| Link | What you get |
|------|----------------|
| **[Open interactive dashboard](https://nabilshehadeh.github.io/trafficanomaly/)** | Graph, histogram and heatmap in the browser (no install). *Enable GitHub Pages from `/docs` to use this URL.* |
| **[Open in Colab](https://colab.research.google.com/github/nabilshehadeh/trafficanomaly/blob/main/Traffic_Anomaly_Interactive_Demo.ipynb)** | Full pipeline plus interactive graph, histogram and heatmap in the notebook. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabilshehadeh/trafficanomaly/blob/main/Traffic_Anomaly_Interactive_Demo.ipynb)

The **dashboard** (`docs/index.html`) and the **notebook** both include:
- **Interactive graph** — traffic volume over time (packets & anomalies)
- **Interactive histogram** — packet length and anomaly score distributions
- **Interactive heatmap** — feature correlation matrix

*To enable the dashboard on GitHub: **Settings → Pages → Source**: Deploy from branch, branch `main`, folder `/docs`. Then the dashboard is at `https://<username>.github.io/trafficanomaly/`. You can also open `docs/index.html` locally in a browser.*

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
